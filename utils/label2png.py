# -*- coding: utf-8 -*-
import os.path as osp
import shutil
import os
from tqdm import tqdm
import cv2


# 主要可以把原始的红色图像转换为黑白的二值图像、

def gt2png(folder_path="C:/Users/OEDan/Desktop/unet/skin/Training_GT", save_folder="C:/Users/OEDan/Desktop/unet/skin/Training_Labels"):
    # folder_path = osp.join(sys_path, "ISBI2016_ISIC_Part1_Training_GroundTruth")
    # save_folder = osp.join(sys_path, "labels")
    if osp.isdir(save_folder):
        # remove
        shutil.rmtree(save_folder)
        # new
        os.makedirs(save_folder)
    else:
        # print(save_folder)
        os.makedirs(save_folder)
    images = os.listdir(folder_path)
    # print(images)
    with tqdm(total=len(images)) as pbar:
        for image in images:
            image_name = image.split(".")[0]
            src_path = osp.join(folder_path, image)
            # img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            img = cv2.imread(src_path, 0)
            img[img > 0] = 255
            save_path = osp.join(save_folder, image_name + ".png")
            cv2.imwrite(save_path, img)
            pbar.update(1)
    print("label convert done")


if __name__ == '__main__':
    gt2png()
