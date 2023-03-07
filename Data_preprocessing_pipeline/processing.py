import numpy as np
import h5py
from utils import *
import numpy as np
import io
from pysteps import motion as motion
import cv2
from satpy import Scene
from pyresample import create_area_def

my_area = create_area_def('my_area', {'proj': 'stere', 'lat_0': 90, 'lon_0': 0, 'lat_ts': 60, 'a': 6378.14, 'b': 6356.75, 'x_0': 0, 'y_0': 0},
                          width=700, height=765, area_extent=(0.0, -4412.208871, 700.795124, -3646.970898), units='m'
                          )


def gaug_frame_preprocessed(file_content_byte: bytes) -> np.array:
    try:
        # read bytes
        f = io.BytesIO(file_content_byte)
        hdf = h5py.File(f, 'r')

        # radar data
        radar_img = np.array(hdf.get('image1/image_data')[:], dtype='float32')
        out_of_image = hdf['image1']['calibration'].attrs['calibration_out_of_image']
        date = read_h5_date(hdf['overview'].attrs['product_datetime_start'])

        # load mask
        mask = np.ones((765, 700), dtype='float32')
        mask[radar_img == out_of_image] = 0

        # Set pixels out of image to 0
        radar_img[radar_img == out_of_image] = 0
        radar_img[radar_img == radar_img[0][0]] = 0

        # convert to mm/h Hidde Leijnse(KNMI) approved
        radar_img = (radar_img / 100) * 12

        # clip
        np.clip(radar_img, 0, 100, out=radar_img)

        # remove small scale
        kernel = np.ones((2, 2), np.float32)
        radar_img = np.array(cv2.morphologyEx(
            radar_img, cv2.MORPH_OPEN, kernel), dtype='float32')

        return (radar_img[np.newaxis, ..., np.newaxis], mask, date)
    except:
        print("Error: could not read radar data")
        return (np.zeros((1, 765, 700, 1)), np.zeros((765, 700)), None)


def prec_frame_preprocessed(file_content_byte: bytes) -> np.array:
    try:
        # read bytes
        f = io.BytesIO(file_content_byte)
        hdf = h5py.File(f, 'r')

        # radar data
        radar_img = np.array(hdf.get('image1/image_data')[:], dtype='float32')
        out_of_image = hdf['image1']['calibration'].attrs['calibration_out_of_image']
        date = read_h5_date(hdf['overview'].attrs['product_datetime_start'])

        # load mask
        mask = np.ones((765, 700), dtype='float32')
        mask[radar_img == out_of_image] = 0

        # Set pixels out of image to 0
        radar_img[radar_img == out_of_image] = 0
        radar_img[radar_img == radar_img[0][0]] = 0

        return (radar_img[np.newaxis, ..., np.newaxis], mask, date)
    except:
        print("Error: could not read radar data")
        return (np.zeros((1, 765, 700, 1)), np.zeros((765, 700)), None)


def echo_frame_preprocessed(file_content_byte: bytes) -> np.array:
    try:
        # read bytes
        f = io.BytesIO(file_content_byte)
        hdf = h5py.File(f, 'r')

        # radar data
        radar_img = np.array(hdf.get('image1/image_data')[:], dtype='float32')
        out_of_image = hdf['image1']['calibration'].attrs['calibration_out_of_image']
        date = read_h5_date(hdf['overview'].attrs['product_datetime_start'])

        # load mask
        mask = np.ones((765, 700), dtype='float32')
        mask[radar_img == out_of_image] = 0

        # Set pixels out of image to 0
        radar_img[radar_img == out_of_image] = 0
        radar_img[radar_img == radar_img[0][0]] = 0

        # radar_img = radar_img * mask
        return (radar_img[np.newaxis, ..., np.newaxis], mask, date)
    except:
        print("Error: could not read radar data")
        return (np.zeros((1, 765, 700, 1)), np.zeros((765, 700)), None)


def motion_fields_create(frame: np.array) -> np.array:
    try:
        # compute motion field
        V1 = np.array(motion.lucaskanade.dense_lucaskanade(
            frame, interp_method='rbfinterp2d'), dtype='float32')  # , interp_method='rbfinterp2d'
        V1 = np.moveaxis(V1, 0, -1)
        V1 = np.around(V1, decimals=2)

        return V1[np.newaxis, ...]
    except:
        print("Error: could not create motion fields")
        return (np.zeros((1, 765, 700, 2)))


def satellite_preprocessed(filenames) -> np.array:
    try:
        scn = Scene(reader='seviri_l1b_hrit', filenames=filenames)
        scn.load(['HRV',
                  'IR_108',
                  'VIS006',
                  'WV_062',
                  'WV_073'])
        new_scn = scn.resample(my_area, resampler='nearest')

        # get data
        date = read_h5_date_sat(scn['WV_073'].attrs['start_time'])
        sat_img = np.empty((1, 765, 700, 5))
        #sat_img[...,0] = new_scn['HRV'].values
        #sat_img[...,1] = new_scn['IR_108'].values
        #sat_img[...,2] = new_scn['VIS006'].values
        #sat_img[...,3] = new_scn['WV_062'].values
        #sat_img[...,4] = new_scn['WV_073'].values

        return (sat_img, date)
    except:
        print("Error: could not read radar data")
        return (np.zeros((1, 765, 700, 5)), None)
