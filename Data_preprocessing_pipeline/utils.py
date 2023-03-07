from pathlib import Path
import yaml
import os
import numpy as np
from datetime import datetime, timedelta


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def read_yaml(file_path) -> str:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def make_dirs(list_dir: list) -> None:
    for l in list_dir:
        l.mkdir(parents=True, exist_ok=True)


def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size >> 20


def get_data_dict(data: tuple):
    try:
        Data = {
            "conditional_data": data[0],
            "target_data": data[1],
            "mask": data[2],
            "start_date": data[3],
        }
    except:
        print("Data could not be loaded to dictionary")
    return Data


def read_h5_date(date):
    date = date.decode('utf-8')
    decoded_date = datetime.strptime(date, '%d-%b-%Y;%H:%M:%S.%f')
    decoded_date = decoded_date.replace(second=0, microsecond=0)
    if decoded_date.minute % 5 != 0:
        decoded_date = roundTime(decoded_date)
    return decoded_date

def read_h5_date_sat(date):
    decoded_date = date.replace(second=0, microsecond=0)
    if decoded_date.minute % 5 != 0:
        decoded_date = roundTime(decoded_date)
    return decoded_date

def read_date(date):
    date = date.decode('utf-8')
    decoded_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return decoded_date


def roundTime(decoded_date):
    if decoded_date.minute % 5 >= 3:
        return decoded_date + timedelta(minutes=5) - timedelta(minutes=decoded_date.minute % 5)
    else:
        return decoded_date - timedelta(minutes=decoded_date.minute % 5)
