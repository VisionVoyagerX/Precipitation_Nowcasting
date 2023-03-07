from pathlib import Path
import tarfile
from processing import *
import datetime
import time


class Iterator_custom():
    """Custom iterator designed to handle missing files. The instance of this class is able to be iterated. 
    Iterations stops when no more files are available. Also the iteration can be stopped and continued in 
    order to handle any missing from the calling function.
    """

    def __init__(self, dir, tar_list, preprocess_func):
        self.tar_iter = iter([dir/tar for tar in tar_list])
        self.open_file = None  # tarfile.open(self.list_tar[0], "r")
        self.member_iter = None  # iter([member for member in self.file])

        self.hold_f = False

        self.last_frame = None
        self.last_mask = None
        self.last_date = None
        self.time = 0

        self.preprocess_func = preprocess_func

    def stop(self):
        self.hold_f = True

    def start(self):
        self.hold_f = False

    def __iter__(self):
        return self

    def __next__(self):
        temp = time.time()
        if not self.hold_f:
            if not self.open_file:
                tar_file = next(self.tar_iter)  # if not returns stopiteration
                self.open_file = tarfile.open(tar_file, "r")
                self.member_iter = iter([member for member in self.open_file])

            file = next(self.member_iter, None)
            if file:
                self.last_frame, self.last_mask, self.last_date = self.preprocess_func(
                    self.open_file.extractfile(file.name).read())
                self.time += time.time() - temp
                return (self.last_frame, self.last_mask, self.last_date)
            else:
                self.open_file = None
                self.member_iter = None
                self.time += time.time() - temp
                return self.__next__()
        else:
            self.time += time.time() - temp
            return (self.last_frame, self.last_mask, self.last_date)
        raise StopIteration


if __name__ == "__main__":
    gauge_f = [Path(
        'Data\downloaded\precipitation_gauge\RAD25_OPER_R___TARRRT__L2__20181228T080500_20181229T080000_0001.tar'),
        Path(
        'Data\downloaded\precipitation_gauge\RAD25_OPER_R___TARRRT__L2__20181229T080500_20181230T080000_0001.tar')]

    prec_f = [Path(
        'Data\downloaded\precipitation\RAD25_OPER_R___TARPCP__L2__20181228T000000_20181229T000000_0001.tar'),
        Path(
        'Data\downloaded\precipitation\RAD25_OPER_R___TARPCP__L2__20181229T000000_20181230T000000_0001.tar')]

    echo_f = [Path(
        'Data\downloaded\echo_top\RAD25_OPER_R___TARETH__L2__20181228T000000_20181229T000000_0001.tar'),
        Path(
        'Data\downloaded\echo_top\RAD25_OPER_R___TARETH__L2__20181229T000000_20181230T000000_0001.tar')]

    iterator_g = Iterate(gauge_f)
    iterator_p = Iterate(prec_f)
    iterator_e = Iterate(echo_f)

    for gaug_h5df_byte, prec_h5df_byte, ech_h5df_byte in zip(iterator_g, iterator_p, iterator_e):
        '''gaug_processed, _, date_g = prec_frame_preprocessed(g_h5df_byte)
        prec_processed, _, date_p = prec_frame_preprocessed(p_h5df_byte)
        ech_processed, _, _ = echo_frame_preprocessed(e_h5df_byte)'''
        gaug, _, date_g = prec_frame_preprocessed(gaug_h5df_byte)
        prec, _, date_p = prec_frame_preprocessed(prec_h5df_byte)
        ech, _, date_e = prec_frame_preprocessed(ech_h5df_byte)
        if date_g > date_p:
            #gaug_processed = prec_processed
            #date_g = date_p
            delta = date_g - date_p
            #print(delta.seconds // (5 * 60))
            if delta.seconds // (5 * 60) < 2:
                print('Missing data. Replacing with precipitation values')
                gaug = prec
                date_g = date_p
                iterator_g.hold_file()
            else:
                print('more than 2 days are missing. frame has been discarded')
                iterator_g.hold_file()
                continue
        else:
            iterator_g.unhold_file()
        print('g_date: ', date_g, ", date_p: ", date_p)
'''with tarfile.open(tar_file, "r") as file:
            for member in file:
                # open without saving to disk
                yield (file.extractfile(member.name).read(), member.name)'''
