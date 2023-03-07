import asyncio
from distutils.command.config import config
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path
import argparse
import time
from utils import *

import requests
from requests import Session


logging.basicConfig(filename='log_files/download.log',
                    filemode='a', format='%(asctime)s %(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))


def download_dataset_file(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
    directory: str,
    overwrite: bool,
) -> Tuple[bool, str]:
    """ function provide from https://developer.dataplatform.knmi.nl/example-scripts """

    # if a file from this dataset already exists, skip downloading it.
    file_path = Path(directory, filename).resolve()
    if not overwrite and file_path.exists():
        logger.info(f"Dataset file '{filename}' was already downloaded.")
        return True, filename

    endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    get_file_response = session.get(endpoint)

    # retrieve download URL for dataset file
    if get_file_response.status_code != 200:
        logger.warning(f"Unable to get file: {filename}")
        logger.warning(get_file_response.content)
        return False, filename

    # use download URL to GET dataset file. We don't need to set the 'Authorization' header,
    # The presigned download URL already has permissions to GET the file contents
    download_url = get_file_response.json().get("temporaryDownloadUrl")
    download_dataset_file_response = requests.get(download_url)

    if download_dataset_file_response.status_code != 200:
        logger.warning(f"Unable to download file: {filename}")
        logger.warning(download_dataset_file_response.content)
        return False, filename

    # write dataset file to disk
    file_path.write_bytes(download_dataset_file_response.content)

    logger.info(f"Downloaded dataset file '{filename}'")
    return True, filename


def list_dataset_files(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    params: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    """ function provide from https://developer.dataplatform.knmi.nl/example-scripts """
    logger.info(f"Retrieve dataset files with query params: {params}")

    list_files_endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"
    list_files_response = session.get(list_files_endpoint, params=params)

    if list_files_response.status_code != 200:
        raise Exception("Unable to list initial dataset files")

    try:
        list_files_response_json = list_files_response.json()
        dataset_files = list_files_response_json.get("files")
        dataset_filenames = list(
            map(lambda x: x.get("filename"), dataset_files))
        return dataset_filenames, list_files_response_json
    except Exception as e:
        logger.exception(e)
        raise Exception(e)


async def main(params, args, rad=None):
    """ function provide from https://developer.dataplatform.knmi.nl/example-scripts """
    api_key = params[0]
    dataset_name = params[1]
    dataset_version = params[2]
    base_url = params[3]
    download_directory = params[4]

    # When set to True, if a file with the same name exists the output is written over the file.
    # To prevent unnecessary bandwidth usage, leave it set to False.
    overwrite = False

    # Make sure to send the API key with every HTTP request
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    # Verify that the download directory exists
    if not Path(download_directory).is_dir() or not Path(download_directory).exists():
        raise Exception(
            f"Invalid or non-existing directory: {download_directory}")

    filenames = []

    start_after_filename = " "
    max_keys = 500

    # Use the API to get a list of all dataset filenames
    while True:
        # Retrieve dataset files after given filename
        dataset_filenames, response_json = list_dataset_files(
            session,
            base_url,
            dataset_name,
            dataset_version,
            {"maxKeys": f"{max_keys}", "startAfterFilename": start_after_filename},
        )

        # Store filenames
        filenames += dataset_filenames

        # If the result is not truncated, we retrieved all filenames
        is_truncated = response_json.get("isTruncated")
        if not is_truncated:
            logger.info("Retrieved names of all dataset files")
            break

        start_after_filename = dataset_filenames[-1]

    if args == 'precipitation' or args == 'precipitation_gauge' or args == 'echo_top':
        filenames = filter_dates(filenames, args)
    else:
        raise Exception("Something went wrong")

    logger.info(f"Number of files to download: {len(filenames)}")
    loop = asyncio.get_event_loop()

    # Allow up to 20 separate threads to download dataset files concurrently
    executor = ThreadPoolExecutor()  # max_workers=20
    futures = []

    # Create tasks that download the dataset files
    for dataset_filename in filenames:
        # Create future for dataset file
        future = loop.run_in_executor(
            executor,
            download_dataset_file,
            session,
            base_url,
            dataset_name,
            dataset_version,
            dataset_filename,
            download_directory,
            overwrite,
        )
        futures.append(future)

    # # Wait for all tasks to complete and gather the results
    future_results = await asyncio.gather(*futures)
    logger.info(f"Finished '{dataset_name}' dataset download")

    failed_downloads = list(filter(lambda x: not x[0], future_results))

    if len(failed_downloads) > 0:
        logger.warning("Failed to download the following dataset files:")
        logger.warning(list(map(lambda x: x[1], failed_downloads)))


def filter_dates(filenames: list, args: str, rad=None) -> list:
    """We do choose the files to download manually by slicing a list. 
    We do not provide the option for the user to specify the download dates 
    since it is computationally expensive.    

    Args:
        filenames (list): list of filenames to be downloaded
        args (str): command line arguments

    Raises:
        Exception: catch any error, such as filenames is empty.

    Returns:
        list: filtered list of filenames to be downloaded
    """
    filenames_filtered = []

    if args == 'precipitation_gauge':
        #filenames_filtered = filenames[9:19]
        filenames_filtered = filenames[13:]
    elif args == 'precipitation':
        #filenames_filtered = filenames[4000:4010]
        filenames_filtered = filenames[4004:]
    elif args == 'echo_top':
        # example name:
        #filenames_filtered = filenames[4000:4010]
        filenames_filtered = filenames[4004:]

    if filenames_filtered:
        return filenames_filtered
    else:
        raise Exception(
            "Something went wrong in the filtering process. Check if filenames are empty")


def download(args: str, cfg: dict) -> None:
    """function to manage the arguments passed in the script, download, and save to directory.

    Args:
        args (string): command line argument .
        cfg (dict): dictionary that contains paths to all directories.

    Raises:
        Exception: when Input argument is incorrect.
    """

    # access info
    base_url = cfg['dataplatform']['base_url']

    if args == "precipitation" or args == "precipitation_gauge" or args == "echo_top":
        # make dirs
        dir = Path(cfg['download_dir'][args])
        make_dirs([dir])

        # dataset info
        api_key = cfg['credentials']['bulk_key'][args]
        dataset_name = cfg['dataset_identification'][args]['dataset_name']
        dataset_version = cfg['dataset_identification'][args]['dataset_version']
        download_directory = cfg['download_dir'][args]
        params = (api_key, dataset_name, dataset_version,
                  base_url, download_directory)
        # download
        asyncio.run(main(params, args))

        # print size
        print(str(args) + " directory size is: " +
              str(get_size(cfg['download_dir'][args])) + " MB")
    elif args == "all":
        args_all = ['precipitation_gauge', 'precipitation', 'echo_top']
        for arg in args_all:
            download(arg, cfg)  # call recursively
    else:
        raise Exception(
            "Arguments passed, not correct. Available inputs: precipitation, " +
            "echo_top. Check that your spelling is correct")


if __name__ == "__main__":  # for 1 day data: execution time: -> 112.57340574264526 seconds | size: 37 MB
    """
        Script used to download datasets from KNMI Dataplatform. To execute this script 
        you have to provide the --dataset argument with the available values being: 
        precipitation, echo_top, all
    """
    # get the start time
    st = time.time()

    # Argument parser
    parser = argparse.ArgumentParser(
        description='Script used to download datasets from KNMI Dataplatform. - make sure to change ' +
                    'the api key in the config files')
    parser.add_argument('--dataset', type=str, required=True,
                        help='provide the name of the dataset you want to download(precipitation, precipitation_gauge, echo_top or all)')
    args = parser.parse_args()

    # download_dataset
    cfg = read_yaml(Path('configs/dataset_config.yml'))
    download(args.dataset, cfg)

    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
