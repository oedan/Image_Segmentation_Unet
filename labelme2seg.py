#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import argparse
import glob
import math
import json
import os
import os.path as osp
import shutil
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2


def json2png(json_folder, png_save_folder):
    if osp.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)
    os.makedirs(png_save_folder)
    json_files = os.listdir(json_folder)
    for json_file in json_files:
        json_path = osp.join(json_folder, json_file)
        os.system("labelme_json_to_dataset {}".format(json_path))
        label_path = osp.join(json_folder, json_file.split(".")[0] + "_json/label.png")
        png_save_path = osp.join(png_save_folder, json_file.split(".")[0] + ".png")
        label_png = cv2.imread(label_path, 0)
        label_png[label_png > 0] = 255
        cv2.imwrite(png_save_path, label_png)
        # shutil.copy(label_path, png_save_path)
        # break


if __name__ == '__main__':
    # !!!!你的json文件夹下只能有json文件不能有其他文件
    json2png(json_folder="testdata/jsons/", png_save_folder="testdata/labels/")
