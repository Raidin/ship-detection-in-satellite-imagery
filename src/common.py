#!/usr/bin/python
"""
A collection of commonly used public methods
Create at : 2018. 12. 27
Writer : jihunjung
"""

import os
from PIL import Image

def check_if_isfile(file_path):
    return os.path.isfile(file_path)

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    return

def Numpy2Image(arr, save_dir):
    # Transpose to Image Type
    # arr shape :: h x w x c
    result_image = Image.fromarray(arr)
    result_image.save(save_dir, 'PNG')


if __name__ == '__main__':
    main()