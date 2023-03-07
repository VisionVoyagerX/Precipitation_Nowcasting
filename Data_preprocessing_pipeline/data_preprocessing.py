import matplotlib.pyplot as plt
import os
from utils import *
from pathlib import Path
import numpy as np
import time
import argparse
import math
import time
import concurrent.futures
from tfrecord_shards import Nowcasting_tfrecord
from numpy.lib.stride_tricks import sliding_window_view
from processing import *
from custom_iterator import Iterator_custom
from custom_iterator_satellite import Iterator_satellite

st_0 = 0  # entire process
# st_1 = 0  # gaug preprocess
# st_2 = 0  # prec preprocess
# st_3 = 0  # echo top
st_4 = 0  # motion fields
st_5 = 0  # windows
st_6 = 0  # crop
st_7 = 0  # save
st_8 = 0  # rest


def preprocess_split_window_crop(in_dir: tuple, out_dir: tuple, tf_record_params: tuple, windowing_params: tuple, cropping_params: tuple, arg, tf_start=0) -> None:
    """takes tar files as input -> split dataset -> untar -> preprocess -> window in training, validation and test window directories"""

    tr_val_te_files = split_by_date(in_dir)

    # for train, validation and test
    for i, (files, out_d, tf_record_params, win_params, crop_params) in enumerate(zip(tr_val_te_files, out_dir, tf_record_params, windowing_params, cropping_params)):
        if arg == i:
            windowing_cropping(
                in_dir, files, out_d, tf_record_params, win_params, crop_params, tf_start)


def split_by_date(in_dir: tuple) -> tuple:
    """split in_dir in training validation and test files"""
    gaug_files = os.listdir(in_dir[0])
    prec_files = os.listdir(in_dir[1])
    echo_files = os.listdir(in_dir[2])

    gaug_files.sort()
    prec_files.sort()
    echo_files.sort()

    # FIXME these values should be changed
    tr_te_split = 10
    step = 4

    # FIXME add satellite
    tr_files = (gaug_files[:tr_te_split],
                prec_files[:tr_te_split],
                echo_files[:tr_te_split])
    val_files = (tr_files[0][0::step],
                 tr_files[1][0::step],
                 tr_files[2][0::step],
                 )
    #del tr_files[0][::step]
    #del tr_files[1][::step]
    #del tr_files[2][::step]
    te_files = (gaug_files[tr_te_split:],
                prec_files[tr_te_split:],
                echo_files[tr_te_split:])

    print('here', len(tr_files[0]))
    return (tr_files, val_files, te_files)


def windowing_cropping(in_dir: tuple, in_files: tuple, out_dir: Path, tf_record_params: tuple,  win_params: tuple, crop_params: tuple, tf_start: int = 0):
    def windowing_cropping_inside(load_list, window_size, stride, dates_list, tf_record, crop_params):
        global st_5
        temp = time.time()
        windows_list, start_dates_list = extract_windows_trick(
            load_list, window_size, stride, np.array(dates_list))
        st_5 += time.time() - temp
        # for window, start_date in zip(windows_list, start_dates_list):
        cropping_and_save(tf_record,
                          windows_list, crop_params, start_dates_list)

    # FIXME experiment with this one
    load_number = 300

    window_size, stride = win_params

    load_max_time = calc_max_time(load_number, window_size, stride)
    max_time = load_max_time - window_size

    dates_list = []
    hist_prec_list = []
    load_list = np.empty((load_max_time, 765, 700, 6), dtype='float32')
    hist_prec = np.empty((0, 765, 700), dtype='float32')

    global st_0
    st_0 = 0  # entire process
    # st_1 = 0  # gaug preprocess
    # st_2 = 0  # prec preprocess
    # st_3 = 0  # echo top
    global st_4
    st_4 = 0  # motion fields
    global st_5
    st_5 = 0  # windows
    global st_6
    st_6 = 0  # crop
    global st_7
    st_7 = 0  # save
    global st_8
    st_8 = 0  # rest
    # tf record for saving data with shards
    tf_filename, max_samples = tf_record_params
    tf_record = Nowcasting_tfrecord(
        out_dir=out_dir, filename=tf_filename, max_samples=max_samples, tf_start=tf_start)

    iterator_g = Iterator_custom(
        in_dir[0], in_files[0], gaug_frame_preprocessed)
    iterator_p = Iterator_custom(
        in_dir[1], in_files[1], prec_frame_preprocessed)
    iterator_e = Iterator_custom(
        in_dir[2], in_files[2], echo_frame_preprocessed)
    '''iterator_s = Iterator_satellite(
        in_dir[3], satellite_preprocessed)'''
    index = 0

    st_0_temp = time.time()
    # for every pair of h5 files
    for (gaug_processed, g_mask, date_g), (prec_processed, p_mask, date_p), (ech_processed, e_mask, date_e) in zip(iterator_g, iterator_p, iterator_e):

        iter_all_data = [[iterator_g, gaug_processed, g_mask, date_g], [iterator_p, prec_processed, p_mask, date_p],
                         [iterator_e, ech_processed, e_mask, date_e]]
        """[iterator_s, sat_processed, s_mask, date_s]]"""

        if date_g != date_p or date_p != date_e:
            iter_all_data.sort(key=lambda item: item[3], reverse=True)

            if (iter_all_data[0][3] - iter_all_data[-1][3]).seconds // (5 * 60) <= 1:
                # print('Only one data missing. Replacing...')
                for i in range(len(iter_all_data)-1):
                    # missing row
                    if iter_all_data[i][3] > iter_all_data[-1][3]:
                        iter_all_data[i][1] = iter_all_data[i][0].last_frame
                        iter_all_data[i][2] = iter_all_data[i][0].last_mask
                        iter_all_data[i][3] = iter_all_data[-1][3]

                        iter_all_data[i][0].stop()
                    # no missing row
                    elif iter_all_data[i][3] == iter_all_data[-1][3]:
                        iter_all_data[i][0].start()
                    iter_all_data[-1][0].start()
                pass
            elif (iter_all_data[0][3] - iter_all_data[-1][3]).seconds // (5 * 60) > 1:
                # print('More than one data is missing')
                # run if len > window_size and then clear
                if index > window_size:
                    # print('Processing remainders')
                    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as motion_executor:
                        results = motion_executor.map(
                            motion_fields_create, hist_prec_list)
                        motion_list = [result for result in results]

                    for i, (ind) in enumerate(range(load_max_time - len(motion_list), load_max_time)):
                        load_list[ind, :, :, 3:5] = motion_list[i]
                    st_4 += time.time() - st_4_temp
                    load_list[..., 3:5] *= load_list[..., [-1]]
                    load_list = load_list[:index, ...]
                    windowing_cropping_inside(
                        load_list, window_size, stride, dates_list, tf_record, crop_params)
                # print('Clearning memory')
                dates_list = []
                hist_prec_list = []
                load_list = np.empty(
                    (load_max_time, 765, 700, 6), dtype='float32')
                index = 0
                # else clear
                for i in range(len(iter_all_data)-1):
                    if iter_all_data[i][3] > iter_all_data[-1][3]:
                        iter_all_data[i][0].stop()
                    elif iter_all_data[i][3] == iter_all_data[-1][3]:
                        iter_all_data[i][0].start()
                    iter_all_data[-1][0].start()
                continue
        else:
            iterator_g.start()
            iterator_p.start()
            iterator_e.start()

        # keep last precipitation frames
        hist_prec = np.concatenate(
            (hist_prec, iter_all_data[0][1][..., 0]), axis=0)
        if hist_prec.shape[0] >= 2:
            hist_prec_list.append(hist_prec)
            hist_prec = hist_prec[1:, ...]

            mask_processed = iter_all_data[0][2] * \
                iter_all_data[1][2] * iter_all_data[2][2]

            mask_processed = mask_processed[np.newaxis, ..., np.newaxis]
            iter_all_data[0][1] *= mask_processed
            iter_all_data[1][1] *= mask_processed
            iter_all_data[2][1] *= mask_processed
            #iter_all_data[3][1] *= mask_processed

            load_list[index, :, :, 0:1] = iter_all_data[0][1]
            load_list[index, :, :, 1:2] = iter_all_data[1][1]
            load_list[index, :, :, 2:3] = iter_all_data[2][1]
            #load_list[index, :, :, 5:10] = iter_all_data[3][1]
            load_list[index, :, :, 5:6] = mask_processed
            dates_list.append(iter_all_data[0][3])
            index += 1
        else:
            continue

        if index >= load_max_time:

            st_4_temp = time.time()

            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as motion_executor:
                results = motion_executor.map(
                    motion_fields_create, hist_prec_list)
                motion_list = [result for result in results]

            for i, (ind) in enumerate(range(load_max_time - len(motion_list), load_max_time)):
                load_list[ind, :, :, 3:5] = motion_list[i]
            st_4 += time.time() - st_4_temp
            load_list[..., 3:5] *= load_list[..., [-1]]

            windowing_cropping_inside(
                load_list, window_size, stride, dates_list, tf_record, crop_params)

            hist_prec_list = []
            temp = np.empty((load_max_time, 765, 700, 6), dtype='float32')
            load_t = load_list[max_time + stride:]
            index = len(load_t)
            temp[0:-max_time - stride] = load_t
            load_list = temp

            dates_list = dates_list[max_time + stride:]

    if index >= window_size:  # remainders
        st_4_temp = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as motion_executor:
            results = motion_executor.map(
                motion_fields_create, hist_prec_list)
            motion_list = [result for result in results]

        for i, (ind) in enumerate(range(load_max_time - len(motion_list), load_max_time)):
            load_list[ind, :, :, 3:5] = motion_list[i]
        st_4 += time.time() - st_4_temp
        load_list[..., 3:5] *= load_list[..., [-1]]
        load_list = load_list[:index, ...]
        windowing_cropping_inside(
            load_list, window_size, stride, dates_list, tf_record, crop_params)
    tf_record.close()

    st_0 = time.time() - st_0_temp

    st_1 = iterator_g.time
    st_2 = iterator_p.time
    st_3 = iterator_e.time

    st_8 = st_0 - st_1 - st_2 - st_3 - st_4 - st_5 - st_6 - st_7
    processes = ['entire', 'gaug', 'prec', 'ech',
                 'mot', 'win', 'crop', 'save', 'rest']
    seconds = [st_0, st_1, st_2, st_3, st_4,
               st_5, st_6, st_7, st_8]

    for p, s in zip(processes, seconds):
        print(p, ": ", s)
    """plt.bar(processes, seconds)
    plt.ylabel('seconds')
    plt.xlabel("Time spend for each process")
    plt.savefig(str(time.time()) + "_" + 'times.png')
    plt.close()"""


def cropping_and_save(tf_record, window_list: list, crop_params: tuple, start_date_list: str) -> None:
    # read params
    crop_size, spatial_offset, random_offset, s, m, q_min = crop_params

    global st_6
    global st_7

    if crop_size:  # training , validation
        st_6_temp = time.time()
        C = crop_size[0] * crop_size[1] * window_list[0].shape[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor_crop:
            results = [
                executor_crop.submit(crop_paral, window, C, s, spatial_offset, q_min, m, start_date, crop_size) for window, start_date in zip(window_list, start_date_list)]

            final_results = [f.result() for f in results]

        st_6 += time.time() - st_6_temp
        st_7_temp = time.time()
        for (final_crops, prob, start_date) in final_results:
            for crop, prob in zip(final_crops, prob):
                tf_record.write_images_to_tfr_long(
                    crop[:4, ...], crop[4:, ..., [0]], crop[..., -1:].astype(bool), str(start_date), prob=prob.astype('float32'))
        st_7 += time.time() - st_7_temp
    else:  # test
        st_7_temp = time.time()
        window = np.pad(window, [(0, 0), (2, 1), (2, 2),
                        (0, 0)], mode='constant', constant_values=(0))

        tf_record.write_images_to_tfr_long(
            window[:4, ...], window[4:, ..., [0]], window[..., -1:].astype(bool), str(start_date))
        st_7 += time.time() - st_7_temp
    return


def crop_paral(window, C, s, spatial_offset, q_min, m, start_date, crop_size):
    crop = window[..., [0]]
    # we do not devide by s since it is 1
    # x_nc_sat = ne.evaluate('1 - exp(-crop)')
    x_nc_sat = 1 - np.exp(- crop / s)
    q_n_sat = sliding_window_view(x_nc_sat, (crop.shape[0], crop_size[0], crop_size[1], 1))[
        :, ::spatial_offset, ::spatial_offset, :]
    q_n = np.minimum(1, q_min + (m/C) *
                     np.sum(q_n_sat, axis=(-4, -3, -2, -1)))
    rand = np.random.uniform(0, 1, (q_n.shape))
    passed = q_n > rand
    probs = q_n[passed]
    sliding_windows = sliding_window_view(window, (window.shape[0], crop_size[0], crop_size[1], 6))[
        :, ::spatial_offset, ::spatial_offset, :]
    final_crops = sliding_windows[passed]

    return final_crops, probs, start_date


def calc_max_time(load_number: int, window_size: int, stride: int) -> int:
    """function to compute the max length of the data, so that when performing windowing the last window
    matches the last timestep"""
    # compute the output dimension of the windowed timeseries
    dim_n = ((load_number - window_size)/stride) + 1
    if dim_n.is_integer():
        return load_number
    # if it is a float that means that we need to reduce the max len size
    else:
        n_dim_n = math.floor(dim_n)
        max_time = ((n_dim_n - 1) * stride) + window_size
        return max_time


def extract_windows_trick(array: tuple, window_size: int, stride: int, dates_list: tuple) -> tuple:
    array_windows = sliding_window_view(array, (window_size, 765, 700, 6))[
        ::stride, 0, 0, 0, ...]
    dates_list = sliding_window_view(dates_list, (window_size))[::stride]

    return (array_windows, list(dates_list[:, 0]))


if __name__ == "__main__":
    # get the start time
    st = time.time()

    # Argument parser
    parser = argparse.ArgumentParser(
        description='Dataset preprocessing(dataset should already be downloaded)' +
        'preprocess all data  -> generate motion fields -> windowing entire dataset -> save to pickles per example')
    parser.add_argument('--set', type=str, required=True)
    parser.add_argument('--tf_start', type=int, required=False)
    args = parser.parse_args()

    if args.set == 'train':
        arg = 0
    elif args.set == 'validation':
        arg = 1
    elif args.set == 'test':
        arg = 2
    else:
        raise Exception(
            "argument provided is not correct(options: train, validation, test")
    if args.tf_start:
        tf_start = args.tf_start
    else:
        tf_start = 0
    # read and make directory paths
    cfg = read_yaml(Path('configs/dataset_config.yml'))
    prec_gauge_download_dir = Path(cfg['download_dir']['precipitation_gauge'])
    prec_download_dir = Path(cfg['download_dir']['precipitation'])
    echo_download_dir = Path(cfg['download_dir']['echo_top'])
    sat_download_dir = Path(cfg['download_dir']['satellite'])
    train_dir = Path(cfg['final_dir']['train'])
    validation_dir = Path(cfg['final_dir']['validation'])
    test_dir = Path(cfg['final_dir']['test'])
    make_dirs([train_dir, validation_dir, test_dir])

    # Parameters
    # Format -> tuples:(train, validation, test)
    save_dirs = (train_dir, validation_dir, test_dir)
    window_size = (14, 24, 24)
    stride = (1, 4, 4)
    crop_size = ((256, 256), (256, 256), None)
    spatial_offset = (32, 256, None)
    random_offset = (True, False, False)
    s = (1, 1, None)
    # 0.1
    m = (1., 4.4, None)
    # (2*(10**(-4))
    q_min = (2*(10**(-6)), 5*(10**(-4)), None)

    window_params = ((window_size[0], stride[0]),
                     (window_size[1], stride[1]),
                     (window_size[2], stride[2]))

    crop_params = ((crop_size[0], spatial_offset[0], random_offset[0], s[0], m[0], q_min[0]),
                   (crop_size[1], spatial_offset[1],
                    random_offset[1], s[1], m[1], q_min[1]),
                   (crop_size[2], spatial_offset[2], random_offset[2], s[2], m[2], q_min[2]))
    tf_params = (("train", 1), ("validation", 1), ("test", 1))

    preprocess_split_window_crop(
        (prec_gauge_download_dir, prec_download_dir, echo_download_dir), save_dirs, tf_params, window_params, crop_params, arg, tf_start)
    # , prec_download_dir, echo_download_dir
    '''uncomment this line in to delete all files except train val and test and reduce storage disk space consumption'''
    # shutil.rmtree(Path(data_temp))

    # get the execution time
    print('Execution time:', (time.time() - st)/60., 'minutes')
