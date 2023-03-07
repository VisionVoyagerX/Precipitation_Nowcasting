from pathlib import Path
import tarfile
from processing import *
import datetime
import time
from glob import glob
import tempfile


class Iterator_satellite():
    """Custom iterator designed to handle missing files. The instance of this class is able to be iterated. 
    Iterations stops when no more files are available. Also the iteration can be stopped and continued in 
    order to handle any missing from the calling function.
    """

    def __init__(self, dir, preprocess_func):
        self.tar_iter = iter(glob(dir + '/*'))
        # self.open_file = None  # tarfile.open(self.list_tar[0], "r")
        self.day_tmpdirname = None
        self.member_iter = None  # iter([member for member in self.file])
        self.file_inner = None

        self.hold_f = False

        self.last_frame = None
        self.last_mask = np.ones((765, 700))
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
            if not self.day_tmpdirname:
                tar_file = next(self.tar_iter)  # if not returns stopiteration
                self.day_tmpdirname = tempfile.TemporaryDirectory(dir=".")
                my_tar = tarfile.open(tar_file, "r")
                my_tar.extractall(self.day_tmpdirname.name)
                my_tar.close()

                self.file_inner = glob(self.day_tmpdirname.name + '/*')[0]
                self.member_iter = iter(glob(self.file_inner + '/*'))

            file_m = next(self.member_iter, None)
            min_tmpdirname = tempfile.TemporaryDirectory(dir=self.file_inner)
            my_tar_inner = tarfile.open(file_m, "r")
            my_tar_inner.extractall(min_tmpdirname.name)
            my_tar_inner.close()
            filenames = glob(min_tmpdirname.name + '/*')
            if filenames:
                self.last_frame, self.last_date = self.preprocess_func(
                    filenames)
                self.time += time.time() - temp
                min_tmpdirname.cleanup()
                return (self.last_frame, self.last_mask, self.last_date)
            else:
                self.file_inner = None
                self.member_iter = None
                self.day_tmpdirname.cleanup()
                self.day_tmpdirname = None
                self.time += time.time() - temp
                return self.__next__()
        else:
            self.time += time.time() - temp
            return (self.last_frame, self.last_mask, self.last_date)
        raise StopIteration


if __name__ == "__main__":
    sat_a = [Path('Data\downloaded\satellite\msg_rss_hrit_201808_09-002.tar')]

    iterator_s = Iterator_satellite(sat_a, satellite_preprocessed)

    for (data, date) in iterator_s:
        print(data.shape)
        print(date)
