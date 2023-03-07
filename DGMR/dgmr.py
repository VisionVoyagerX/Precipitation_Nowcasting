import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
import time


class DGMR(tf.keras.Model):
    def __init__(self, lead_time=90) -> None:
        super(DGMR, self).__init__()
        self.strategy = None
        self.global_step = 0
        self.generator_obj = Generator(
            lead_time=lead_time)
        self.discriminator_obj = Discriminator()

    def compile(self, gen_optimizer, disc_optimizer, gen_loss, disc_loss):
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss

    def fit(self, dataset,  steps=2, callbacks=[]):
        train_writer = callbacks[0]
        ckpt_manager = callbacks[1]
        ckpt = callbacks[2]
        # tf.profiler.experimental.start(callbacks[3])

        disc_loss_l = []
        gen_loss_l = []

        '''if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')'''
        dataset = iter(dataset)
        num_batches = self.global_step

        '''start_profiling_step = 40
        stop_profiling_step = 90'''

        for step in range(steps):
            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):

            '''if step == start_profiling_step:
              tf.profiler.experimental.start(logdir=callbacks[3])
            if step == stop_profiling_step:
              tf.profiler.experimental.stop(save=True)'''

            batch_inputs1, batch_targets1, targ_mask1, _ = next(dataset)
            batch_inputs2, batch_targets2, targ_mask2, _ = next(dataset)

            temp_time = time.time()
            gen_loss, disc_loss = self.distributed_train_step(
                batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2)

            if num_batches < 50:
                tf.print("Time per epoch: ", time.time() - temp_time)

            # Loss
            disc_loss_l.append(disc_loss)
            gen_loss_l.append(gen_loss)

            num_batches += 1

            if step and (step % 1 == 0):  # default 100
                tf.print("Gen Loss: ", gen_loss.numpy(),
                         " Disc Loss: ", disc_loss.numpy())

            if step and (step % 2000 == 0):
                ckpt_save_path = ckpt_manager.save()

            if step and (step % 5 == 0):
                with train_writer.as_default():
                    tf.summary.scalar("Gen_loss", gen_loss, step=step)
                    tf.summary.scalar("Disc_loss", disc_loss, step=step)

        # tf.profiler.experimental.stop()

        return gen_loss_l, disc_loss_l

    # @tf.function
    def distributed_train_step(self, batch_inputs1, batch_targets1, targ_mask1,  batch_inputs2, batch_targets2, targ_mask2):
        per_replica_g_losses, per_replica_d_losses = self.strategy.run(self.train_step,
                                                                       args=(batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2))

        total_g_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_g_losses, axis=0)
        total_d_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_d_losses, axis=0)

        return total_g_loss, total_d_loss

    def train_step(self, batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2):
        self.disc_step(batch_inputs1, batch_targets1, targ_mask1)
        disc_loss = self.disc_step(batch_inputs2, batch_targets2, targ_mask2)

        #self.gen_step(batch_inputs2, batch_targets2, targ_mask2)
        #self.gen_step(batch_inputs2, batch_targets2, targ_mask2)
        gen_loss = self.gen_step(batch_inputs2, batch_targets2, targ_mask2)

        return gen_loss, disc_loss

    def disc_step(self, batch_inputs, batch_targets, targ_mask):
        with tf.GradientTape() as disc_tape:
            batch_predictions = self.generator_obj(batch_inputs)
            '''batch_predictions = tf.where(
                tf.equal(targ_mask, True), batch_predictions, -1/32)'''
            gen_sequence = tf.concat(
                [batch_inputs[..., :1], batch_predictions], axis=1)
            real_sequence = tf.concat(
                [batch_inputs[..., :1], batch_targets], axis=1)
            concat_inputs = tf.concat(
                [real_sequence, gen_sequence], axis=0)

            concat_outputs = self.discriminator_obj(concat_inputs)

            score_real, score_generated = tf.split(
                concat_outputs, 2, axis=0)
            disc_loss = self.disc_loss(score_generated, score_real)

        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator_obj.trainable_variables)

        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator_obj.trainable_variables))
        return disc_loss

    def gen_step(self, batch_inputs, batch_targets, targ_mask):
        with tf.GradientTape() as gen_tape:
            num_samples_per_input = 6
            gen_samples = [self.generator_obj(batch_inputs)
                           for _ in range(num_samples_per_input)]

            grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),
                                                  batch_targets)
            gen_sequences = [tf.concat([batch_inputs[..., :1], x], axis=1)
                             for x in gen_samples]
            real_sequence = tf.concat(
                [batch_inputs[..., :1], batch_targets], axis=1)

            generated_scores = []
            for g_seq in gen_sequences:
                concat_inputs = tf.concat([real_sequence, g_seq], axis=0)
                concat_outputs = self.discriminator_obj(concat_inputs)
                score_real, score_generated = tf.split(
                    concat_outputs, 2, axis=0)
                generated_scores.append(score_generated)

            gen_disc_loss = self.gen_loss(tf.concat(generated_scores, axis=0))
            gen_loss = gen_disc_loss + 10.0 * grid_cell_reg

        gen_grads = gen_tape.gradient(
            gen_loss, self.generator_obj.trainable_variables)

        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator_obj.trainable_variables))
        return gen_loss


def grid_cell_regularizer(generated_samples, batch_targets):
    gen_mean = tf.reduce_mean(generated_samples, axis=0)
    weights = tf.clip_by_value(batch_targets, 0.0, 2.0)
    loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights)
    return loss
