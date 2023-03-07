#import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_pipeline import Dataset
from pathlib import Path
from dgmr import DGMR
from losses import Loss_hing_disc, Loss_hing_gen
import os
import matplotlib.pyplot as plt
from utils import *
import argparse
from tensorflow import keras

#os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# os.environ["NCCL_DEBUG"]="WARN"

# TF_GPU_ALLOCATOR="cuda_malloc_async"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Argument parser
parser = argparse.ArgumentParser(
    description='Train DGMR model by providing configuration file name')
parser.add_argument('--config', type=str, required=True,
                    help='provide the name of the configuration file you want to run')
args = parser.parse_args()

cfg = read_yaml(Path('configs/' + args.config + '.yml'))

MODEL_NAME = cfg['model_identification']['model_name']
MODEL_VERSION = cfg['model_identification']['model_version']
ROOT = get_project_root()
CHECKPOINT_DIR = ROOT / 'Checkpoints' / \
    (str(MODEL_NAME) + '_v' + str(MODEL_VERSION))
make_dirs([CHECKPOINT_DIR])


training_steps = cfg['model_params']['steps']

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(18)
# tf.config.set_soft_device_placement(True)

#gpu_devices_list = tf.config.list_physical_devices('GPU')

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


train_data = Dataset(Path('Data/train'), batch_size=GLOBAL_BATCH_SIZE)


# tensorboard_callback = keras.callbacks.TensorBoard(
#    log_dir="tb_callback_dir", histogram_freq=1
# )
train_writer = tf.summary.create_file_writer(
    str(ROOT / "logs" / (str(MODEL_NAME) + '_v' + str(MODEL_VERSION)) / "train/"))

prof_dir = str(ROOT / "logs" / (str(MODEL_NAME) +
                                '_v' + str(MODEL_VERSION)) / "profiler/")
#profiler_writer = tf.summary.create_file_writer(prof_dir)

# INIT MODEL
disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
gen_optimizer = Adam(learning_rate=5E-5, beta_1=0.0, beta_2=0.999)
loss_hinge_gen = Loss_hing_gen()
loss_hinge_disc = Loss_hing_disc()
with strategy.scope():

    my_model = DGMR(lead_time=5)

    my_model.compile(gen_optimizer, disc_optimizer,
                     loss_hinge_gen, loss_hinge_disc)

    my_model.strategy = strategy

    ckpt = tf.train.Checkpoint(generator=my_model.generator_obj,
                               discriminator=my_model.discriminator_obj,
                               generator_optimizer=my_model.gen_optimizer,
                               discriminator_optimizer=my_model.disc_optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, CHECKPOINT_DIR, max_to_keep=100)

    '''if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')'''


'''my_model.generator_obj.prec_normalizer.adapt(train_data.map(
    lambda x, y, z, d: x[..., :1]).take(100000))'''


train_dist_dataset = strategy.experimental_distribute_dataset(train_data)

my_model.fit(train_dist_dataset, steps=training_steps, callbacks=[
    train_writer, ckpt_manager, ckpt])  # , prof_dir
