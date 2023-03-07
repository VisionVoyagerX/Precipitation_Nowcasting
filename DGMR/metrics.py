import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
import pysteps
import numpy as np
from pysteps.utils.spectral import rapsd
from pysteps.utils import rapsd, transformation
from pysteps.verification.probscores import CRPS


class CSI_score():
    def __init__(self, threshold) -> None:
        self.threshold = threshold

        self.TP = np.zeros(1)
        self.FP = np.zeros(1)
        self.FN = np.zeros(1)

    def update(self, batch_gen, batch_target):
        pred = crop_middle(batch_gen)
        targ = crop_middle(batch_target)
        pred = pred > self.threshold
        targ = targ > self.threshold
        batch_gen_u = tf.unstack(pred, axis=1)
        batch_target_u = tf.unstack(targ, axis=1)

        self.TP = self.TP + np.array([tf.keras.metrics.TruePositives()(pred_temp, targ_temp).numpy()
                                      for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)])
        self.FP = self.FP + np.array([tf.keras.metrics.FalsePositives()(pred_temp, targ_temp).numpy()
                                      for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)])
        self.FN = self.FN + np.array([tf.keras.metrics.FalseNegatives()(pred_temp, targ_temp).numpy()
                                      for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)])

    def result(self):
        return self.TP / (self.TP + self.FP + self.FN)


class BIAS_score():
    def __init__(self, threshold) -> None:
        self.TP = np.zeros(1)
        self.FP = np.zeros(1)
        self.FN = np.zeros(1)
        self.threshold = threshold

    def update(self, batch_gen, batch_target):
        pred = crop_middle(batch_gen)
        targ = crop_middle(batch_target)
        pred = pred > self.threshold
        targ = targ > self.threshold
        batch_gen_u = tf.unstack(pred, axis=1)
        batch_target_u = tf.unstack(targ, axis=1)

        self.TP = self.TP + np.array([tf.keras.metrics.TruePositives()(pred_temp, targ_temp).numpy()
                                      for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)])
        self.FP = self.FP + np.array([tf.keras.metrics.FalsePositives()(pred_temp, targ_temp).numpy()
                                      for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)])
        self.FN = self.FN + np.array([tf.keras.metrics.FalseNegatives()(pred_temp, targ_temp).numpy()
                                      for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)])

    def result(self):
        return (self.TP + self.FP) / (self.TP + self.FN)


class MSE_score():
    def __init__(self) -> None:
        pass

    def __call__(self, batch_gen, batch_target):
        pred = crop_middle(batch_gen)
        targ = crop_middle(batch_target)
        pred_u = tf.unstack(pred, axis=1)
        targ_u = tf.unstack(targ, axis=1)

        mse_score = [tf.reduce_mean(tf.math.square(pred - targ)).numpy()
                     for pred, targ in zip(pred_u, targ_u)]
        return np.array(mse_score)


class Rank_histogram():
    def __init__(self) -> None:
        pass

    def __call__(self, batch_gen_list, batch_target):
        #b_pred_list = tf.unstack(batch_gen_list, axis=1)
        pred_l = [crop_middle(pred) for pred in batch_gen_list]
        pred = tf.stack(pred_l, axis=0)
        targ = crop_middle(batch_target)

        tf.print("pred", pred.shape)
        tf.print("targ", targ.shape)

        rank_scores = np.zeros((pred.shape[0] + 1)).astype('float32')
        counter = 0

        tf.print(pred.shape)
        tf.print(targ.shape)

        for i in range(targ.shape[1]):
            # for j in range(targ.shape[1]):
            rank_scores += pysteps.verification.ensscores.rankhist(
                pred[:, 0, i, ..., 0].numpy(), targ[0, i, ..., 0].numpy())
            counter += 1
        rank_score = rank_scores/counter
        return np.array(rank_score)


class Max_Pool_CRPS():
    def __init__(self, K) -> None:
        T = 18  # FIXME change that to 18
        stride = K // 4
        if stride == 0:
            stride = 1
        self.max_pool_3d = tf.keras.layers.MaxPool3D(
            pool_size=(1, K, K), strides=(1, stride, stride), padding='Valid')

    def __call__(self, b_pred, b_targ):
        pred = [crop_middle(pred) for pred in b_pred]
        targ = crop_middle(b_targ)

        #rand_ind = np.random.randint(low=0, high=len(pred), size=(2))
        pooled_pred = [self.max_pool_3d(pre) for pre in pred]
        pooled_targ = self.max_pool_3d(targ)

        pooled_pred = tf.stack(pooled_pred, axis=1)
        pooled_crps_score = [CRPS(pooled_pred.numpy()[0, :, i, ..., 0],
                                  pooled_targ.numpy()[0, i, ..., 0]) for i in range(pooled_pred.shape[2])]
        # tf.reduce_mean(np.abs(pooled_pred[rand_ind[0]] - pooled_targ) - 0.5 * np.abs(
        # pooled_pred[rand_ind[0]] - pooled_pred[rand_ind[1]]), axis=[0, 2, 3, 4])

        return np.array(pooled_crps_score)


class Avg_Pool_CRPS():
    def __init__(self, K) -> None:
        T = 18  # FIXME change that to 20
        stride = K // 4
        if stride == 0:
            stride = 1
        self.avg_pool_3d = tf.keras.layers.AvgPool3D(
            pool_size=(1, K, K), strides=(1, stride, stride), padding='Valid')

    def __call__(self, b_pred, b_targ):
        pred = [crop_middle(pred) for pred in b_pred]
        targ = crop_middle(b_targ)

        pooled_pred = [self.avg_pool_3d(pre) for pre in pred]
        pooled_targ = self.avg_pool_3d(targ)

        pooled_pred = tf.stack(pooled_pred, axis=1)
        pooled_crps_score = [CRPS(pooled_pred.numpy()[0, :, i, ..., 0],
                                  pooled_targ.numpy()[0, i, ..., 0]) for i in range(pooled_pred.shape[2])]

        return np.array(pooled_crps_score)


class PSD():
    def __init__(self, timestep) -> None:
        T = 18  # FIXME change that to 20
        self.timestep = timestep

    def __call__(self, b_pred):
        pred = crop_middle(b_pred[:, self.timestep: self.timestep+1, ...])

        # Log-transform the data
        R, metadata = transformation.dB_transform(
            pred.numpy()[0, 0, ..., 0], threshold=0.1, zerovalue=-15.0)

        # Assign the fill value to all the Nans
        R[~np.isfinite(R)] = metadata["zerovalue"]
        R_, freq = rapsd(R, fft_method=np.fft, return_freq=True)

        return (R_, freq)
