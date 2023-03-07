from turtle import end_fill
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import time


class Nowcasting_tfrecord():
    # , cond_shapes: tuple = None, tar_shapes: tuple = None
    def __init__(self, out_dir: str = None, filename: str = None, max_samples: int = None, tf_start: int = 0):
        """This class can be used to initialize, save, store and read data.

        Args:
            out_dir (str, optional): directory to save the tf record files. Defaults to None.
            filename (str, optional): filename. Defaults to None.
            cond_shapes (tuple, optional): shape of conditional data. Defaults to None.
            tar_shapes (tuple, optional): shape of target data. Defaults to None.
            max_samples (int, optional): max number of samples per . Defaults to None.
        """

        self.filename = filename
        self.out_dir = out_dir

        #self.cond_shapes = cond_shapes
        #self.tar_shapes = tar_shapes
        self.max_samples = max_samples

        self.first = True
        self.file_index = tf_start
        self.file_count = 0

        self.current_shard_count = 0
        self.current_shard_name = None
        self.writer = None

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value is a tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a floast_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_array(self, array):
        array = tf.io.serialize_tensor(array)
        return array

    def parse_tfr_element(self, element):
        # use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            'window_cond': tf.io.FixedLenFeature([], tf.float32),
            'height_cond': tf.io.FixedLenFeature([], tf.float32),
            'width_cond': tf.io.FixedLenFeature([], tf.float32),
            'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
            'depth_cond': tf.io.FixedLenFeature([], tf.float32),
            'window_targ': tf.io.FixedLenFeature([], tf.float32),
            'height_targ': tf.io.FixedLenFeature([], tf.float32),
            'width_targ': tf.io.FixedLenFeature([], tf.float32),
            'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
            'depth_targ': tf.io.FixedLenFeature([], tf.float32),
            'window_mask': tf.io.FixedLenFeature([], tf.int64),
            'height_mask': tf.io.FixedLenFeature([], tf.int64),
            'width_mask': tf.io.FixedLenFeature([], tf.int64),
            'raw_image_mask': tf.io.FixedLenFeature([], tf.string),
            'depth_mask': tf.io.FixedLenFeature([], tf.int64),
            'start_date': tf.io.FixedLenFeature([], tf.string)
        }

        content = tf.io.parse_single_example(element, data)

        window_cond = content['window_cond']
        height_cond = content['height_cond']
        width_cond = content['width_cond']
        depth_cond = content['depth_cond']
        raw_image_cond = content['raw_image_cond']
        window_targ = content['window_targ']
        height_targ = content['height_targ']
        width_targ = content['width_targ']
        depth_targ = content['depth_targ']
        raw_image_targ = content['raw_image_targ']
        window_mask = content['window_mask']
        height_mask = content['height_mask']
        width_mask = content['width_mask']
        depth_mask = content['depth_mask']
        raw_image_mask = content['raw_image_mask']
        start_date = content['start_date']

        # get our 'feature'-- our image -- and reshape it appropriately
        feature_cond = tf.io.parse_tensor(raw_image_cond, out_type=tf.float32)
        feature_cond = tf.reshape(
            feature_cond, shape=[window_cond, height_cond, width_cond, depth_cond])

        feature_targ = tf.io.parse_tensor(raw_image_targ, out_type=tf.float32)
        feature_targ = tf.reshape(
            feature_targ, shape=[window_targ, height_targ, width_targ, depth_targ])

        feature_mask = tf.io.parse_tensor(raw_image_mask, out_type=tf.bool)
        feature_mask = tf.reshape(
            feature_mask, shape=[window_mask, height_mask, width_mask, depth_mask])

        feature_date = tf.io.parse_tensor(start_date, out_type=tf.string)
        return (feature_cond, feature_targ, feature_mask, feature_date)

    def parse_tfr_element_with_prob(self, element):
        # use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            'window_cond': tf.io.FixedLenFeature([], tf.float32),
            'height_cond': tf.io.FixedLenFeature([], tf.float32),
            'width_cond': tf.io.FixedLenFeature([], tf.float32),
            'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
            'depth_cond': tf.io.FixedLenFeature([], tf.float32),
            'window_targ': tf.io.FixedLenFeature([], tf.float32),
            'height_targ': tf.io.FixedLenFeature([], tf.float32),
            'width_targ': tf.io.FixedLenFeature([], tf.float32),
            'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
            'depth_targ': tf.io.FixedLenFeature([], tf.float32),
            'window_mask': tf.io.FixedLenFeature([], tf.int64),
            'height_mask': tf.io.FixedLenFeature([], tf.int64),
            'width_mask': tf.io.FixedLenFeature([], tf.int64),
            'raw_image_mask': tf.io.FixedLenFeature([], tf.string),
            'depth_mask': tf.io.FixedLenFeature([], tf.int64),
            'prob': tf.io.FixedLenFeature([], tf.string),
            'start_date': tf.io.FixedLenFeature([], tf.string)
        }

        content = tf.io.parse_single_example(element, data)

        window_cond = content['window_cond']
        height_cond = content['height_cond']
        width_cond = content['width_cond']
        depth_cond = content['depth_cond']
        raw_image_cond = content['raw_image_cond']
        window_targ = content['window_targ']
        height_targ = content['height_targ']
        width_targ = content['width_targ']
        depth_targ = content['depth_targ']
        raw_image_targ = content['raw_image_targ']
        window_mask = content['window_mask']
        height_mask = content['height_mask']
        width_mask = content['width_mask']
        depth_mask = content['depth_mask']
        raw_image_mask = content['raw_image_mask']
        prob = content['prob']
        start_date = content['start_date']

        # get our 'feature'-- our image -- and reshape it appropriately
        feature_cond = tf.io.parse_tensor(raw_image_cond, out_type=tf.float32)
        feature_cond = tf.reshape(
            feature_cond, shape=[window_cond, height_cond, width_cond, depth_cond])

        feature_targ = tf.io.parse_tensor(raw_image_targ, out_type=tf.float32)
        feature_targ = tf.reshape(
            feature_targ, shape=[window_targ, height_targ, width_targ, depth_targ])

        feature_mask = tf.io.parse_tensor(raw_image_mask, out_type=tf.bool)
        feature_mask = tf.reshape(
            feature_mask, shape=[window_mask, height_mask, width_mask, depth_mask])

        feature_prob = tf.io.parse_tensor(prob, out_type=tf.float32)

        feature_date = tf.io.parse_tensor(start_date, out_type=tf.string)
        return (feature_cond, feature_targ, feature_mask, feature_prob, feature_date)

    def write_images_to_tfr_long(self, cond, targ, mask, date, prob=None):

        if self.first or self.current_shard_count >= self.max_samples:
            if self.writer:
                # close the previous file
                self.file_index += 1
                self.current_shard_count = 0
                self.writer.close()
            # open a new file
            self.current_shard_name = self.out_dir / "{}_{}.tfrecords".format(
                self.file_index+1, self.filename)
            options = tf.io.TFRecordOptions(compression_type="GZIP")

            self.writer = tf.io.TFRecordWriter(
                str(self.current_shard_name), options)  #
            self.first = False

        # create the required Example representation
        if prob:
            out = self.parse_single_image_with_prob(
                cond=cond, targ=targ, mask=mask, prob=prob, start_date=date)
        else:
            out = self.parse_single_image(
                cond=cond, targ=targ, mask=mask, start_date=date)

        self.writer.write(out.SerializeToString())
        self.current_shard_count += 1
        self.file_count += 1

        return self.file_count

    def parse_single_image(self, cond, targ, mask, start_date):
        # define the dictionary -- the structure -- of our single example
        data = {
            'window_cond': self._float_feature(cond.shape[0]),
            'height_cond': self._float_feature(cond.shape[1]),
            'width_cond': self._float_feature(cond.shape[2]),
            'depth_cond': self._float_feature(cond.shape[3]),
            'raw_image_cond': self._bytes_feature(self.serialize_array(cond)),
            'window_targ': self._float_feature(targ.shape[0]),
            'height_targ': self._float_feature(targ.shape[1]),
            'width_targ': self._float_feature(targ.shape[2]),
            'depth_targ': self._float_feature(targ.shape[3]),
            'raw_image_targ': self._bytes_feature(self.serialize_array(targ)),
            'window_mask': self._int64_feature(mask.shape[0]),
            'height_mask': self._int64_feature(mask.shape[1]),
            'width_mask': self._int64_feature(mask.shape[2]),
            'depth_mask': self._int64_feature(mask.shape[3]),
            'raw_image_mask': self._bytes_feature(self.serialize_array(mask)),
            'start_date': self._bytes_feature(self.serialize_array(start_date)),
        }
        # create an Example, wrapping the single features
        out = tf.train.Example(features=tf.train.Features(feature=data))

        return out

    def parse_single_image_with_prob(self, cond, targ, mask, prob, start_date):

        # define the dictionary -- the structure -- of our single example
        data = {
            'window_cond': self._float_feature(cond.shape[0]),
            'height_cond': self._float_feature(cond.shape[1]),
            'width_cond': self._float_feature(cond.shape[2]),
            'depth_cond': self._float_feature(cond.shape[3]),
            'raw_image_cond': self._bytes_feature(self.serialize_array(cond)),
            'window_targ': self._float_feature(targ.shape[0]),
            'height_targ': self._float_feature(targ.shape[1]),
            'width_targ': self._float_feature(targ.shape[2]),
            'depth_targ': self._float_feature(targ.shape[3]),
            'raw_image_targ': self._bytes_feature(self.serialize_array(targ)),
            'window_mask': self._int64_feature(mask.shape[0]),
            'height_mask': self._int64_feature(mask.shape[1]),
            'width_mask': self._int64_feature(mask.shape[2]),
            'depth_mask': self._int64_feature(mask.shape[3]),
            'raw_image_mask': self._bytes_feature(self.serialize_array(mask)),
            'prob': self._bytes_feature(self.serialize_array(prob)),
            'start_date': self._bytes_feature(self.serialize_array(start_date)),
        }
        # create an Example, wrapping the single features
        out = tf.train.Example(features=tf.train.Features(feature=data))

        return out

    def get_dataset_large(self, tfr_dir: str, pattern: str, has_prob=False):
        files = glob.glob(tfr_dir+'/'+pattern, recursive=False)

        # create the dataset
        dataset = tf.data.TFRecordDataset(
            files, compression_type='GZIP')  # , compression_type='GZIP'  # , num_parallel_reads=tf.data.AUTOTUNE
        # pass every single feature through our mapping function
        if has_prob:
            dataset = dataset.map(
                self.parse_tfr_element_with_prob
            )
        else:
            dataset = dataset.map(
                self.parse_tfr_element
            )

        return dataset

    def close(self):
        self.writer.close()
        print(f"Wrote {self.file_count} elements to TFRecord")
        return self.file_count
