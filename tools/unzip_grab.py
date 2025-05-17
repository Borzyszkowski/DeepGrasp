""" Extracts the downloaded ZIP files of GRAB """

import argparse
import os
import shutil

from tools.utils import makepath


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='GRAB-unzip')
    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data (all zip files)')
    parser.add_argument('--extract-path', default=None, type=str,
                        help='The path to the folder to extract GRAB to')
    return parser.parse_args()


def unpack_grab(zip_path, unzip_path=None):
    """ Unzips the downloaded GRAB dataset """
    all_zips = [f for f in os.walk(zip_path)]

    if unzip_path is None:
        unzip_path = zip_path + '_unzipped'

    makepath(unzip_path)

    for directory, folder, files in all_zips:
        for file in files:
            children = file.split('__')[:-1]

            extract_dir = os.path.join(unzip_path, *children)
            zip_name = os.path.join(directory, file)
            makepath(extract_dir)
            print(f'unzipping:\n'
                  f'{zip_name}\n'
                  f'to :\n'
                  f'{extract_dir}\n'
                  )
            shutil.unpack_archive(zip_name, extract_dir, 'zip')

    print(f'Unzipping finished! GRAB dataset saved in {extract_dir}')


if __name__ == '__main__':
    args = parse_args()
    unpack_grab(args.grab_path, args.extract_path)
    