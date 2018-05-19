#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hashlib
import io

from PIL import Image


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
IMG_MODEL_FOLDER = os.path.join(os.path.dirname(__file__), 'img_model')
img_value_dict = dict()


def get_num(img_data, index_skip_info):
    f = io.BytesIO(img_data)
    img = Image.open(f)
    width, height = img.size
    counter = 0
    last_width = 0
    for skip_w, skip_x in index_skip_info:
        counter += 1
        skip_w = int(skip_w)
        skip_x = int(skip_x)
        box = (skip_x, 0, skip_x + skip_w, height)
        new_img = img.crop(box)
        if counter == 1:
            end_img = Image.new('RGB', (100, new_img.size[1]))
        end_img.paste(new_img, (last_width, 0))
        last_width += new_img.size[0]

    return get_value_from_img(img=end_img)


def get_value_from_img(fp=None, img=None):
    if not fp and not img:
        raise Exception('param error')
    if not img and fp:
        img = Image.open(fp)
    img = img.convert('RGB')
    img_data = img.load()
    img_width, img_height = img.size
    for x in range(img_width):
        for y in range(img_height):
            if img_data[x, y] != WHITE:
                img_data[x, y] = WHITE
            else:
                img_data[x, y] = BLACK
    small_imgs = split_img(img, img_data, img_width, img_height)
    return get_value_from_small_imgs(small_imgs)


def get_value_from_small_imgs(small_imgs):
    global img_value_dict
    value = []
    for img in small_imgs:
        key = get_md5(img)
        value.append(img_value_dict[key])
    return "".join(value)


def split_img(img, img_data, img_width, img_height):
    imgs = []
    split_info = []
    left = right = top = bottom = 0
    y_set = set()
    for x in range(img_width):
        all_is_white = True
        for y in range(img_height):
            if img_data[x, y] == WHITE:
                continue
            all_is_white = False
            if not left:
                left = x
            y_set.add(y)
        if all_is_white and left and not right:
            right = x
            top = min(y_set)
            bottom = max(y_set)
            split_info.append((left, right, top, bottom))
            left = right = top = bottom = 0
            y_set = set()
    for left, right, top, bottom in split_info:
        box = (left, top - 1, right, bottom + 1)
        new_img = img.crop(box)
        imgs.append(new_img)
    return imgs


def get_md5(img):
    content_list = []
    img = img.convert('RGB')
    img_data = img.load()
    img_width, img_height = img.size
    for x in range(img_width):
        for y in range(img_height):
            content = 'x:{0},y:{1},{2}'.format(x, y, img_data[x, y])
            content_list.append(content)
    return hashlib.md5("".join(content_list).encode()).hexdigest()


def _load_imgs():
    global img_value_dict
    file_name_list = os.listdir(IMG_MODEL_FOLDER)
    for file_name in file_name_list:
        value = file_name.split('.')[0]
        file_path = os.path.join(IMG_MODEL_FOLDER, file_name)
        img = Image.open(file_path)
        key = get_md5(img)
        img_value_dict[key] = value


_load_imgs()