# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import imghdr
import cv2
import random
import numpy as np
import json
# import torch

def cal_IoU(label,pred):
    sq1 = (label["points"][2][0]-label["points"][0][0])*(label["points"][2][1]-label["points"][0][1])
    sq2 = (pred["points"][2][0]-pred["points"][0][0])*(pred["points"][2][1]-pred["points"][0][1])
    # print(sq1,sq2)
    x = -max(label["points"][0][0],pred["points"][0][0])+min(label["points"][2][0],pred["points"][2][0])
    if x<0:
        x = 0 
    y = -max(label["points"][0][1],pred["points"][0][1])+min(label["points"][2][1],pred["points"][2][1])
    if y < 0:
        y = 0   
    # print(x*y)
    # print(x,y)
    IoU = x*y / (sq1+sq2-x*y)
    return IoU

def json2label(json_file):
    annotations_list = []
    with open(json_file,encoding="UTF-8") as f:
        json_file = json.loads(f.read())
        image_name = json_file[0]["image"]
        for ann in json_file[0]["annotations"]:
            label = ann["label"]
            crop_idx = [[int(ann['coordinates']["x"]-ann['coordinates']["width"]/2),int(ann['coordinates']["y"]-ann['coordinates']["height"]/2)],
                        [int(ann['coordinates']["x"]+ann['coordinates']["width"]/2),int(ann['coordinates']["y"]-ann['coordinates']["height"]/2)],
                        [int(ann['coordinates']["x"]+ann['coordinates']["width"]/2),int(ann['coordinates']["y"]+ann['coordinates']["height"]/2)],
                        [int(ann['coordinates']["x"]-ann['coordinates']["width"]/2),int(ann['coordinates']["y"]+ann['coordinates']["height"]/2)]]
            annotations_list.append({"transcription": label, "points": crop_idx})
    annotations_list = json.dumps(annotations_list,ensure_ascii = False)
    return f"{annotations_list}\n"

def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


def get_check_global_params(mode):
    check_params = ['use_gpu', 'max_text_length', 'image_shape', \
                    'image_shape', 'character_type', 'loss_type']
    if mode == "train_eval":
        check_params = check_params + [ \
            'train_batch_size_per_card', 'test_batch_size_per_card']
    elif mode == "test":
        check_params = check_params + ['test_batch_size_per_card']
    return check_params


def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file,label_file=None):
    imgs_lists = []
    label_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    if os.path.isfile(img_file) and _check_image_file(img_file):
        if label_file:
            if os.path.isfile(label_file):
                imgs_lists.append(img_file)
                label_lists.append(label_file)
        else:
            imgs_lists.append(img_file)

    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if label_file:
                single_label_file = os.path.splitext(single_file)[0]+".json"
                label_path = os.path.join(label_file, single_label_file )
            if os.path.isfile(file_path) and _check_image_file(file_path):
                if label_file:
                    if os.path.isfile(label_path):
                        imgs_lists.append(file_path)
                        label_lists.append(label_path)
                else:
                    imgs_lists.append(file_path)
                    
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    label_lists = sorted(label_lists)

    if label_file:
        return imgs_lists, label_lists
    return imgs_lists


def check_and_read(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logger = logging.getLogger('ppocr')
            logger.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:] in ['pdf']:
        import fitz
        from PIL import Image
        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.pageCount):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.getPixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False


def load_vqa_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
    old_lines = [line.strip() for line in lines]
    lines = ["O"]
    for line in old_lines:
        # "O" has already been in lines
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    labels = ["O"]
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
