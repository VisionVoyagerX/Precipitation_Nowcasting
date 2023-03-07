from data_pipeline import Dataset
from pathlib import Path
from metrics import *
from utils import *
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_pipeline import Dataset
from pathlib import Path
from dgmr import DGMR
from losses import Loss_hing_disc, Loss_hing_gen
import numpy as np
import argparse


def csi_report(scores, step, writer):
    with writer.as_default():
        tf.summary.image(
            "CSI score (Perfect score: 1)",
            plot_csi_score(*scores),
            step=step
        )


def bias_report(scores, step, writer):
    with writer.as_default():
        tf.summary.image(
            "BIAS score (Perfect score: 1)",
            plot_bias_score(*scores),
            step=step
        )


def mse_report(score, step, writer):
    with writer.as_default():
        tf.summary.image(
            "MSE score (Perfect score 0)",
            plot_mse_score(score),
            step=step
        )


def rankHist_report(score, step, writer):
    with writer.as_default():
        tf.summary.image(
            "Rank Histogram",
            plot_rankhist_score(score),
            step=step
        )


def max_pool_crps_report(scores, step, writer):
    with writer.as_default():
        tf.summary.image(
            "Max Pool CRPS score (Perfect score: 0)",
            plot_max_pool_crps_score(*scores),
            step=step
        )


def avg_pool_crps_report(scores, step, writer):
    with writer.as_default():
        tf.summary.image(
            "Average Pool CRPS score (Perfect score: 0)",
            plot_avg_pool_crps_score(*scores),
            step=step
        )


def psd_report(scores, step, writer):
    with writer.as_default():
        tf.summary.image(
            "PSD score (match the ground truth",
            psd_score(*scores),
            step=step
        )


def evaluate(dgmr, test_data, writer, dire, num_samples):
    csi_score1_method = CSI_score(1.)
    csi_score4_method = CSI_score(4.)
    csi_score8_method = CSI_score(8.)
    bias_score1_method = BIAS_score(1.)
    bias_score4_method = BIAS_score(4.)
    bias_score8_method = BIAS_score(8.)
    mse_score_method = MSE_score()
    rankHist_score_method = Rank_histogram()
    max_pool_crps1_method = Max_Pool_CRPS(1.)
    max_pool_crps4_method = Max_Pool_CRPS(4.)
    max_pool_crps16_method = Max_Pool_CRPS(16.)
    avg_pool_crps1_method = Avg_Pool_CRPS(1.)
    avg_pool_crps4_method = Avg_Pool_CRPS(4.)
    avg_pool_crps16_method = Avg_Pool_CRPS(16.)
    psd_30_method = PSD(1)
    psd_60_method = PSD(3)
    psd_90_method = PSD(5)

    dataset = iter(test_data)
    step = 0

    batch_counter = 0
    csi_score1, csi_score4, csi_score8 = 0, 0, 0
    bias_score1, bias_score4, bias_score8 = 0, 0, 0
    mse_score_ = 0
    rank_hist_scores = 0
    rank_counter = 0
    max_pool_crps_score_1 = 0
    max_pool_crps_score_4 = 0
    max_pool_crps_score_16 = 0
    avg_pool_crps_score_1 = 0
    avg_pool_crps_score_4 = 0
    avg_pool_crps_score_16 = 0
    psd_score_30 = 0
    psd_score_60 = 0
    psd_score_90 = 0
    freqs = 0

    # @#############################################
    for inputs, targets, _ in dataset:

        #inputs, targets, _ = next(dataset)
        temp = time.time()
        ens_accept = 0.05 > np.random.rand()
        both_accept = 0.2 > np.random.rand()

        if ens_accept:
            gen_seq = [dgmr(inputs, is_training=True)
                       for _ in range(num_samples)]
            gen = gen_seq[np.random.randint(len(gen_seq) - 1)]
        elif both_accept:
            gen = dgmr(inputs, is_training=True)
        else:
            continue

        csi_score1 += csi_score1_method(gen, targets)
        csi_score4 += csi_score4_method(gen, targets)
        csi_score8 += csi_score8_method(gen, targets)

        bias_score1 += bias_score1_method(gen, targets)
        bias_score4 += bias_score4_method(gen, targets)
        bias_score8 += bias_score8_method(gen, targets)

        mse_score_ += mse_score_method(gen, targets)

        if ens_accept:
            max_pool_crps_score_1 += max_pool_crps1_method(gen_seq, targets)
            max_pool_crps_score_4 += max_pool_crps4_method(gen_seq, targets)
            max_pool_crps_score_16 += max_pool_crps16_method(gen_seq, targets)

            avg_pool_crps_score_1 += avg_pool_crps1_method(gen_seq, targets)
            avg_pool_crps_score_4 += avg_pool_crps4_method(gen_seq, targets)
            avg_pool_crps_score_16 += avg_pool_crps16_method(gen_seq, targets)

            if np.random.rand() < (0.3):
                rank_hist_scores += rankHist_score_method(gen_seq, targets)
                rank_counter += 1

        freqs += psd_30_method(gen)[1]
        psd_score_30 += psd_30_method(gen)[0]
        psd_score_60 += psd_60_method(gen)[0]
        psd_score_90 += psd_90_method(gen)[0]

        batch_counter += 1

        print('Time for 1 sample: ', time.time() - temp)

    # @##################################

    csi_score1 = csi_score1 / batch_counter
    csi_score4 = csi_score4 / batch_counter
    csi_score8 = csi_score8 / batch_counter

    bias_score1 = bias_score1 / batch_counter
    bias_score4 = bias_score4 / batch_counter
    bias_score8 = bias_score8 / batch_counter

    mse_score_ = mse_score_ / batch_counter

    if rank_counter > 0:
        rank_hist_scores = rank_hist_scores / rank_counter

    max_pool_crps_score_1 = max_pool_crps_score_1 / batch_counter
    max_pool_crps_score_4 = max_pool_crps_score_4 / batch_counter
    max_pool_crps_score_16 = max_pool_crps_score_16 / batch_counter
    avg_pool_crps_score_1 = avg_pool_crps_score_1 / batch_counter
    avg_pool_crps_score_4 = avg_pool_crps_score_4 / batch_counter
    avg_pool_crps_score_16 = avg_pool_crps_score_16 / batch_counter

    psd_score_30 = psd_score_30 / batch_counter
    psd_score_60 = psd_score_30 / batch_counter
    psd_score_90 = psd_score_30 / batch_counter

    evals = {'CSI1': csi_score1,
             'CSI4': csi_score4,
             'CSI8': csi_score8,
             'BIAS1': bias_score1,
             'BIAS4': bias_score4,
             'BIAS8': bias_score8,
             'MSE': mse_score_,
             'Rank_Hist': rank_hist_scores,
             'MAX_CRPS1': max_pool_crps_score_1,
             'MAX_CRPS4': max_pool_crps_score_4,
             'MAX_CRPS16': max_pool_crps_score_16,
             'AVG_CRPS1': avg_pool_crps_score_1,
             'AVG_CRPS4': avg_pool_crps_score_4,
             'AVG_CRPS16': avg_pool_crps_score_16,
             'PSD30': psd_score_30,
             'PSD60': psd_score_60,
             'PSD90': psd_score_90,
             }

    np.save(dire + 'eval.npy', evals)

    csi_report((csi_score1, csi_score4, csi_score8), step, writer)
    bias_report((bias_score1, bias_score4, bias_score8), step, writer)
    mse_report(mse_score_, step, writer)
    if rank_counter > 0:
        rankHist_report(rank_hist_scores, step, writer)
    max_pool_crps_report((max_pool_crps_score_1, max_pool_crps_score_4, max_pool_crps_score_16),
                         step, writer)
    avg_pool_crps_report((avg_pool_crps_score_1, avg_pool_crps_score_4, avg_pool_crps_score_16),
                         step, writer)

    psd_report((psd_score_30, psd_score_60, psd_score_90, freqs), step, writer)


# Argument parser
parser = argparse.ArgumentParser(
    description='DGMR evaluation')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--checkpoint', type=int, required=True)
args = parser.parse_args()


model_name = args.model
checkpoint = args.checkpoint

test_data = Dataset(Path('Data/test'), batch_size=1,
                    prob=False, shuffle=False)  # no shuffling

val_dir = "./logs/" + str(model_name) + '_v0' + \
    "/evaluation/" + 'checkpoint_' + str(checkpoint) + '/'
val_writer = tf.summary.create_file_writer(
    "./logs/" + str(model_name) + '_v0' + "/evaluation/" + 'checkpoint_' + str(checkpoint))


ROOT = get_project_root()

CHECKPOINT_DIR = ROOT / 'Checkpoints' / \
    (str(model_name) + '_v' + str(0))
print('checkpoint is: ', CHECKPOINT_DIR)

disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
gen_optimizer = Adam(learning_rate=5E-5, beta_1=0.0, beta_2=0.999)
loss_hinge_gen = Loss_hing_gen()
loss_hinge_disc = Loss_hing_disc()
my_model = DGMR(lead_time=30, time_delta=5)
my_model.compile(gen_optimizer, disc_optimizer,
                 loss_hinge_gen, loss_hinge_disc)


ckpt = tf.train.Checkpoint(generator=my_model.generator_obj,
                           discriminator=my_model.discriminator_obj,
                           generator_optimizer=my_model.gen_optimizer,
                           discriminator_optimizer=my_model.disc_optimizer)

ckpt_manager = tf.train.CheckpointManager(
    ckpt, CHECKPOINT_DIR, max_to_keep=10)

if ckpt_manager.latest_checkpoint:
    # ckpt.restore(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.checkpoints[checkpoint])
    print('Latest checkpoint restored!!')

generator = my_model.generator_obj


evaluate(generator, test_data, val_writer, val_dir, num_samples=4)
