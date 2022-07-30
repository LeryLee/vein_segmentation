
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import h5py
import scipy.misc as misc
import re
import glog as log
import functools
import utils.image_utils as image_utils

re_digits = re.compile(r'(\d+)')


def emb_numbers(s):
    pieces = re_digits.split(s)
    if len(pieces) == 7:
        pieces[5::6] = map(int, pieces[5::6])
    else:
        pieces[7::8] = map(int, pieces[7::8])
    return pieces


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # log.info("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    #
    # log.info(np.shape(img))
    # log.info(np.shape(mask))

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


def random_Dataset_loader(img, mask, b_map):

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

    b_map = np.expand_dims(b_map, axis=2)
    b_map = np.array(b_map, np.float32).transpose(2, 0, 1)

    return img, mask, b_map


def default_Dataset_loader(img, mask, b_map):

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

    b_map = np.expand_dims(b_map, axis=2)
    b_map = np.array(b_map, np.float32).transpose(2, 0, 1)

    return img, mask, b_map


def image_reader(img_path, outline_path, vein_path):
    if outline_path is not None:
        img = cv2.imread(img_path)
        outline = cv2.imread(outline_path, cv2.IMREAD_GRAYSCALE)
        vein = cv2.imread(vein_path, cv2.IMREAD_GRAYSCALE)
 
        mask = vein
        index = np.where(outline < 100)
        mask[index] = outline[index]
        # mask = outline + vein
        mask = 255 - mask
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (448, 448), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((img.shape[:2]))
    
    b_map = np.ones(mask.shape)

    return img, mask, b_map


def image_reader_crop_center____(img_path, mask_path):
    crop_size = [3000, 2250]
    if mask_path is not None:
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img[img.shape[0]//2-crop_size[0]//2:img.shape[0]//2+crop_size[0]//2,
                         img.shape[1]//2-crop_size[1]//2:img.shape[1]//2+crop_size[1]//2],
                         (448, 448),
                         interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask[mask.shape[0]//2-crop_size[0]//2:mask.shape[0]//2+crop_size[0]//2,
                          mask.shape[1]//2-crop_size[1]//2:mask.shape[1]//2+crop_size[1]//2],
                          (448, 448),
                          interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img[img.shape[0]//2-crop_size[0]//2:img.shape[0]//2+crop_size[0]//2,
                         img.shape[1]//2-crop_size[1]//2:img.shape[1]//2+crop_size[1]//2],
                         (448, 448),
                         interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((img.shape[:2]))

    return img, mask


def read_datasets(dataset, root_path, mode='train'):

    source_root = os.path.join(root_path, dataset)

    if mode=='train':
        image_list = os.listdir(os.path.join(source_root, 'pretrain'))
        image_root = os.path.join(source_root, 'pretrain')
    elif mode=='retrain':
        image_list = os.listdir(os.path.join(source_root, 'train'))
        image_root = os.path.join(source_root, 'train')
    elif mode=='train_all':
        image_list = os.listdir(os.path.join(source_root, 'train'))
        image_root = os.path.join(source_root, 'train')
    elif mode=='valid':
        image_list = os.listdir(os.path.join(source_root, 'test'))[-20:]
        image_root = os.path.join(source_root, 'test')
    elif mode=='test':
        image_list = os.listdir(os.path.join(source_root, 'test'))
        image_root = os.path.join(source_root, 'test')
    else:
        image_list = os.listdir(os.path.join(source_root, 'all'))
        image_root = os.path.join(source_root, 'all')

    images = list()
    outlines = list()
    veins = list()
    image_names = list()

    num = 0.0
    log.info('mode: {}'.format(mode))
    for image_name in image_list:
        for layer in os.listdir(os.path.join(image_root, image_name)):
            if '背景' in layer:
                image_path = os.path.join(image_root, image_name, layer)
            elif '图层 1' in layer:
                outline_path = os.path.join(image_root, image_name, layer)
            elif '图层 2' in layer:
                pass
            elif '图层 3' in layer:
                vein_path = os.path.join(image_root, image_name, layer)
            else:
                log.info('{} : {} has no data'.format(image_name, layer))

        if mode == 'retrain':
            outline_path = None
            vein_path = None

        images.append(image_path)
        outlines.append(outline_path)
        veins.append(vein_path)
        image_names.append(image_name)

        log.info(f'{image_name}, {str(num)}')
        num += 1.0

    return images, outlines, veins, image_names


def data_cut(img, mask, mask_label, size):

    x_shift = [0, 0, size//2, size//2]
    y_shift = [0, size//2, 0, size//2]

    # fix img,mask and mask_label, map all kind of shift for x and y
    partial_im_cut = functools.partial(image_utils.im_cut,
        img=img, mask=mask, mask_label=mask_label, block_size=size*6, patch_size=size)

    # return (image_blocks, masks_blocks, label_blocks)
    cut_blocks = map(partial_im_cut, x_shift, y_shift)

    # concatenate each image_blocks, masks_blocks and label_blocks
    #result = functools.reduce(image_utils.im_cat_n, cut_blocks)
    result = image_utils.im_cat_n(cut_blocks)
    # log.info(result[0].shape, result[1].shape, result[2].shape)

    image_blocks = result[0]
    masks_blocks = result[1]
    label_blocks = result[2]

    return image_blocks, masks_blocks, label_blocks

def data_frag(img, mask, mask_label, size):
    # blocks to patches

    # fix shift pixels in x and y, map all image blocks
    partial_im_frag = functools.partial(image_utils.im_frag,
        patch_size=size, x_shift=0, y_shift=0)
    single_flag = False
    if len(img.shape) == 3:
        single_flag = True
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask_label = np.expand_dims(mask_label, axis=0)

    # return (image_patches, mask_patches, labels)
    frag_patches = map(partial_im_frag, img, mask, mask_label)

    # concatenate each image_patches, mask_patches and labels
    if single_flag is True:
        result = list(frag_patches)
    else:
        #result = functools.reduce(image_utils.im_cat_n, frag_patches)
        result = image_utils.im_cat_n(frag_patches)
    # log.info(result[0].shape, result[1].shape, result[2].shape)

    image_patches =  np.array(result[0])
    mask_patches = np.array(result[1])
    label_patches = np.array(result[2])

    return image_patches, mask_patches, label_patches

def data_whole_frag(img, mask, mask_label, size):
    # whole image to patches, skip block stage

    x_shift = [0, 0, size // 2, size // 2]
    y_shift = [0, size // 2, 0, size // 2]

    # fix img,mask and mask_label, map all kind of shift for x and y
    partial_im_frag = functools.partial(image_utils.im_frag_map_shift,
                                       img=img, mask=mask, mask_label=mask_label, patch_size=size)

    # return (image_blocks, masks_blocks, label_blocks)
    frag_patches = map(partial_im_frag, x_shift, y_shift)

    # concatenate each image_blocks, masks_blocks and label_blocks
    #result = functools.reduce(image_utils.im_cat_n, frag_patches)
    result = image_utils.im_cat_n(frag_patches)
    # log.info(result[0].shape, result[1].shape, result[2].shape)

    image_patches = result[0]
    mask_patches = result[1]
    label_patches = result[2]
    index_patches = result[3]

    return image_patches, mask_patches, label_patches, index_patches

class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='36_Holly', mode='train', is_random=True):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        self.is_random = is_random
        name_list = ['1_Walnut', '2_SmokeTree', '3_Poplar', '4_OrientalCherry', '5_ChineseRedbud',
                     '6_CrapeMyrtle', '7_Hackberry', '8_CrataegusPinnatifida', '9_VirginiaCreeper',
                     '10_ForsythiaSuspensa', '11_FructusXanthii', '12_Cynanchum', '13_Grape',
                     '14_Hibiscus', '15_MorningGlory', '16_Apricot', '17_ChenopodiumAlbum',
                     '18_PhloxPaniculata', '19_CallistephusChinensis', '20_MapleTree', '21_Amaranth',
                     '22_Honeysuckle', '23_SweetPotato', '24_Cedar', '25_Thistle', '26_MirabilisJalapa',
                     '27_Sycamores', '28_Lilac', '29_Persimmon', '30_Mulberry', '31_SichuanPepper',
                     '32_VitexNegundoVar', '33_MagnoliaDenudata', '34_ChineseRose', '35_Elm', '36_Holly']
        for i in range(len(name_list)):
            name_list[i] = name_list[i] + '_labels'
        assert self.dataset in name_list, "the dataset should be within range"

        self.images, self.outlines, self.veins, self.image_names = read_datasets(self.dataset, self.root, self.mode)
        self.masks = None 
        self.b_maps = None

    def __getitem__(self, index):

        # img, mask = default_DRIVE_loader(self.images[index], self.masks[index])
        is_random = self.is_random

        img, mask, b_map = image_reader(self.images[index], self.outlines[index], self.veins[index])
        
        if self.masks != None:
            mask = self.masks[index]
            b_map = self.b_maps[index]

        if is_random is False:
            #TODO: speed up loading
            img, mask, b_map = default_Dataset_loader(img, mask, b_map)
        else:
            img, mask, b_map = random_Dataset_loader(img, mask, b_map)


        img = torch.tensor(img, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        b_map = torch.tensor(b_map, dtype=torch.float)

        return img, mask, b_map

    def __len__(self):
        # assert len(self.images) == len(self.masks) == len(self.b_maps), 'The number of images must be equal to labels'
        assert len(self.images) == len(self.outlines) == len(self.veins), 'The number of images must be equal to labels'
        return len(self.images)

    def update(self, images, masks, b_maps, is_random):
        # images: images_path
        # masks: [448, 448], binary [0, 255] images
        # b_maps: [448, 448], [0,1] images

        self.images = images
        self.masks = masks
        self.b_maps = b_maps
        self.is_random = is_random