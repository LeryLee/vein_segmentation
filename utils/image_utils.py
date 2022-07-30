import os
import scipy.misc as misc
import shutil
import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F
import functools

#from skimage import morphology

def im_cat_1(tp1, tp2):
    """
    element-wise concatenate numpy array for tp1 and tp2 in same position
    tp1 and tp2 only contain one type data
    :param tp1: data type1
    :param tp2: data type1
    :return:
    """
    result = np.concatenate((tp1,tp2), axis=0)

    return result

def im_cat_n(input):
    """
    element-wise concatenate numpy array for tp1 and tp2 in same position
    :param tp1: (data type1, data type2, data type3)
    :param tp2: (data type1, data type2, data type3)
    :return:
    """
    input = list(input)
    if len(input)==1:
        return input
    n = len(input[0])
    results = []
    for i in range(n):
        tp = [x[i] for x in input]
        tp = np.concatenate(tp, axis=0)
        results.append(tp)
    return results

def im_stack_n(input):
    """
    element-wise concatenate numpy array for tp1 and tp2 in same position
    :param tp1: (data type1, data type2, data type3)
    :param tp2: (data type1, data type2, data type3)
    :return:
    """
    input = list(input)
    if len(input)==1:
        return input
    n = len(input[0])
    results = []
    for i in range(n):
        tp = [x[i] for x in input]
        #tp = np.concatenate(tp, axis=0)
        tp = np.array(tp)
        results.append(tp)
    return results

def im_cut(x_shift, y_shift, img, mask, mask_label, block_size, patch_size):
    """
    Cut the image into several big blocks
    :param img:
    :param mask:
    :param mask_label:
    :param block_size: block_size = patch_size * 6
    :param patch_size: 64
    :param x_shift:
    :param y_shift:
    :param folder:
    :return: image_blocks, mask_blocks, label_blocks
    """
    # init variables
    rows, cols = mask.shape[0:2]
    v_num = math.ceil(rows/block_size)
    h_num = math.ceil(cols/block_size)

    # image_blocks = np.zeros((v_num * h_num, block_size, block_size, 3))
    # mask_blocks = np.zeros((v_num * h_num, block_size, block_size))
    # label_blocks = np.zeros(v_num * h_num)

    image_blocks = []
    mask_blocks = []
    label_blocks = []
    indices = []

    index_seg = 0
    r1, r2 = [x_shift, block_size + x_shift]
    while r2<= rows:
        c1, c2 = [y_shift, block_size + y_shift]
        while c2 <= cols:
            img_seg = img[r1:r2, c1:c2]
            mask_seg = mask[r1:r2, c1:c2]
            label_seg = mask_label[r1:r2, c1:c2]
            # image_blocks[index_seg] = img_seg
            # mask_blocks[index_seg] = mask_seg
            # label_blocks[index_seg] = label_seg

            image_blocks.append(img_seg)
            mask_blocks.append(mask_seg)
            label_blocks.append(label_seg)

            indices.append([r1,r2,c1,c2])

            index_seg, c1, c2 = [index_seg + 1, c1 + block_size - patch_size, c2 + block_size - patch_size]
        r1, r2 = [r1 + block_size - patch_size, r2 + block_size - patch_size]

    image_blocks = np.stack(image_blocks, axis=0)
    mask_blocks = np.stack(mask_blocks, axis=0)
    label_blocks = np.stack(label_blocks, axis=0)
    indices = np.array(indices)

    return image_blocks, mask_blocks, label_blocks, indices


def im_frag(img, mask, mask_label, patch_size, x_shift, y_shift):
    """
    Fragment the image block into several patches
    :param img:
    :param mask:
    :param mask_label:
    :param size:
    :param x_shift:
    :param y_shift:
    :param folder:
    :return:
    """
    # init variables
    rows, cols = mask.shape[0:2]
    v_num = int(rows/patch_size)
    h_num = int(cols/patch_size)

    # image_patches = np.zeros((v_num * h_num, size, size, 3))
    # mask_patches = np.zeros((v_num * h_num, size, size))
    # labels = np.zeros(v_num * h_num)

    image_patches = []
    mask_patches = []
    labels = []
    indices = []

    index_seg = 0
    r1, r2 = [x_shift, patch_size + x_shift]
    while r2<= rows:
        c1, c2 = [y_shift, patch_size + y_shift]
        while c2 <= cols:
            img_seg = img[r1:r2, c1:c2]
            mask_seg = mask[r1:r2, c1:c2]
            mask_label_seg = mask_label[r1:r2, c1:c2]

            # image_patches[index_seg] = img_seg
            # mask_block[index_seg] = mask_seg
            image_patches.append(img_seg)
            mask_patches.append(mask_seg)

            class_0 = np.where(mask_label_seg == 1)
            class_1 = np.where(mask_label_seg == 2)
            class_2 = np.where(mask_label_seg == 3)
            class_3 = np.where(mask_label_seg == 4)

            ratio_0 = len(class_0[0]) / (patch_size * patch_size)
            ratio_1 = len(class_1[0]) / (patch_size * patch_size)
            ratio_2 = len(class_2[0]) / (patch_size * patch_size)
            ratio_3 = len(class_3[0]) / (patch_size * patch_size)

            if ratio_0 > 0.99:
                labels.append(0)
                # labels[index_seg] = 0
            elif ratio_1 > 0:
                if ratio_2 > 0.99:
                    labels.append(2)
                    # labels[index_seg] = 2
                elif ratio_3 > ratio_1:
                    labels.append(3)
                    # labels[index_seg] = 3
                else:
                    labels.append(1)
                    # labels[index_seg] = 1
            elif ratio_2 > 0.96:
                labels.append(2)
                # labels[index_seg] = 2
            else:
                assert ratio_3 > 0, 'no vein'
                labels.append(3)
                # labels[index_seg] = 3

            indices.append([r1, r2, c1, c2])

            index_seg, c1, c2 = [index_seg + 1, c1 + patch_size, c2 + patch_size]
        r1, r2 = [r1 + patch_size, r2 + patch_size]

    image_patches = np.stack(image_patches, axis=0)
    mask_patches = np.stack(mask_patches, axis=0)
    labels = np.stack(labels, axis=0)
    indices = np.array(indices)

    return image_patches, mask_patches, labels, indices

def im_frag_map_shift(x_shift, y_shift, img, mask, mask_label, patch_size):
    """
    Fragment the image block into several patches
    :param img:
    :param mask:
    :param mask_label:
    :param size:
    :param x_shift:
    :param y_shift:
    :param folder:
    :return:
    """
    # init variables
    rows, cols = mask.shape[0:2]
    v_num = int(rows/patch_size)
    h_num = int(cols/patch_size)

    # image_patches = np.zeros((v_num * h_num, size, size, 3))
    # mask_patches = np.zeros((v_num * h_num, size, size))
    # labels = np.zeros(v_num * h_num)

    image_patches = []
    mask_patches = []
    labels = []
    indices = []

    index_seg = 0
    r1, r2 = [x_shift, patch_size + x_shift]
    while r2<= rows:
        c1, c2 = [y_shift, patch_size + y_shift]
        while c2 <= cols:
            img_seg = img[r1:r2, c1:c2]
            mask_seg = mask[r1:r2, c1:c2]
            mask_label_seg = mask_label[r1:r2, c1:c2]

            # image_patches[index_seg] = img_seg
            # mask_block[index_seg] = mask_seg
            image_patches.append(img_seg)
            mask_patches.append(mask_seg)

            class_0 = np.where(mask_label_seg == 1)
            class_1 = np.where(mask_label_seg == 2)
            class_2 = np.where(mask_label_seg == 3)
            class_3 = np.where(mask_label_seg == 4)

            ratio_0 = len(class_0[0]) / (patch_size * patch_size)
            ratio_1 = len(class_1[0]) / (patch_size * patch_size)
            ratio_2 = len(class_2[0]) / (patch_size * patch_size)
            ratio_3 = len(class_3[0]) / (patch_size * patch_size)

            if ratio_0 > 0.99:
                labels.append(0)
                # labels[index_seg] = 0
            elif ratio_1 > 0:
                if ratio_2 > 0.99:
                    labels.append(2)
                    # labels[index_seg] = 2
                elif ratio_3 > ratio_1:
                    labels.append(3)
                    # labels[index_seg] = 3
                else:
                    labels.append(1)
                    # labels[index_seg] = 1
            elif ratio_2 > 0.96:
                labels.append(2)
                # labels[index_seg] = 2
            else:
                assert ratio_3 > 0, 'no vein'
                labels.append(3)
                # labels[index_seg] = 3

            indices.append([r1, r2, c1, c2])

            index_seg, c1, c2 = [index_seg + 1, c1 + patch_size, c2 + patch_size]
        r1, r2 = [r1 + patch_size, r2 + patch_size]

    image_patches = np.stack(image_patches, axis=0)
    mask_patches = np.stack(mask_patches, axis=0)
    labels = np.stack(labels, axis=0)
    indices = np.array(indices)

    return image_patches, mask_patches, labels, indices

def im_frag_map_shift_retrain(x_shift, y_shift, img, mask, mask_label, patch_size):
    """
    Fragment the image block into several patches
    :param img:
    :param mask:
    :param mask_label:
    :param size:
    :param x_shift:
    :param y_shift:
    :param folder:
    :return:
    """
    # init variables
    rows, cols = mask.shape[0:2]
    v_num = int(rows/patch_size)
    h_num = int(cols/patch_size)

    # image_patches = np.zeros((v_num * h_num, size, size, 3))
    # mask_patches = np.zeros((v_num * h_num, size, size))
    # labels = np.zeros(v_num * h_num)

    image_patches = []
    mask_patches = []
    labels = []
    indices = []

    index_seg = 0
    r1, r2 = [x_shift, patch_size + x_shift]
    while r2<= rows:
        c1, c2 = [y_shift, patch_size + y_shift]
        while c2 <= cols:
            img_seg = img[r1:r2, c1:c2]
            mask_seg = mask[r1:r2, c1:c2]
            mask_label_seg = mask_label[r1:r2, c1:c2]

            # image_patches[index_seg] = img_seg
            # mask_block[index_seg] = mask_seg
            image_patches.append(img_seg)
            mask_patches.append(mask_seg)

            class_0 = np.where(mask_label_seg == 1)
            class_1 = np.where(mask_label_seg == 2)
            class_2 = np.where(mask_label_seg == 3)
            class_3 = np.where(mask_label_seg == 4)
            class_non = np.where(mask_label_seg == 0)

            ratio_0 = len(class_0[0]) / (patch_size * patch_size)
            ratio_1 = len(class_1[0]) / (patch_size * patch_size)
            ratio_2 = len(class_2[0]) / (patch_size * patch_size)
            ratio_3 = len(class_3[0]) / (patch_size * patch_size)
            ratio_non = len(class_non[0]) / (patch_size * patch_size)

            if ratio_non < 0.05:
                if ratio_0 > 0.99:
                    labels.append(0)
                    # labels[index_seg] = 0
                elif ratio_1 > 0:
                    if ratio_2 > 0.99:
                        labels.append(2)
                        # labels[index_seg] = 2
                    elif ratio_3 > ratio_1:
                        labels.append(3)
                        # labels[index_seg] = 3
                    else:
                        labels.append(1)
                        # labels[index_seg] = 1
                elif ratio_2 > 0.96:
                    labels.append(2)
                    # labels[index_seg] = 2
                else:
                    # assert ratio_3 > 0, 'no vein'
                    # labels.append(3)
                    if ratio_3 > 0:
                        labels.append(3)
                    else:
                        print('no vein and non label')
                        labels.append(4)
                    # labels[index_seg] = 3
            else:
                labels.append(4)

            indices.append([r1, r2, c1, c2])

            index_seg, c1, c2 = [index_seg + 1, c1 + patch_size, c2 + patch_size]
        r1, r2 = [r1 + patch_size, r2 + patch_size]

    image_patches = np.stack(image_patches, axis=0)
    mask_patches = np.stack(mask_patches, axis=0)
    labels = np.stack(labels, axis=0)
    indices = np.array(indices)

    return image_patches, mask_patches, labels, indices

def get_true_label(x, y, mask_label_gt, mask_label_gt_seg, size):
    class_i = mask_label_gt[x, y] - 1

    class_0 = np.where(mask_label_gt_seg == 1)
    class_1 = np.where(mask_label_gt_seg == 2)
    class_2 = np.where(mask_label_gt_seg == 3)
    class_3 = np.where(mask_label_gt_seg == 4)

    ratio_0 = len(class_0[0]) / (size * size)
    ratio_1 = len(class_1[0]) / (size * size)
    ratio_2 = len(class_2[0]) / (size * size)
    ratio_3 = len(class_3[0]) / (size * size)

    if class_i == 0:
        if ratio_0 > 0.99:
            label_gt = 0
        else:
            label_gt = 4
    elif class_i == 1:
        label_gt = 1
    elif class_i == 2:
        if ratio_2 > 0.99:
            label_gt = 2
        else:
            label_gt = 4
    elif class_i == 3:
        if ratio_1 > 0:
            label_gt = 13
        else:
            label_gt = 3

    else:
        print('class i does not exist!!!')

    return label_gt

def im_frag_map_center_retrain(img, mask, mask_label_gt, mask_label, size):
    # init variables
    rows, cols = mask.shape[0:2]
    v_num = int(rows - size / 2)
    h_num = int(cols - size / 2)
    pointMark = np.zeros(mask_label.shape)

    print(np.where(mask_label == 4))
    print(len(np.where(mask_label == 4)[0]))

    pointMark[np.where(mask_label == 0)] = 1
    pixel_index = np.where(pointMark == 0)

    #     img_patches = np.zeros((v_num * h_num, size, size, 3))
    #     mask_patches = np.zeros((v_num * h_num, size, size))
    #     labels = np.zeros(v_num * h_num)

    img_tmp = []
    mask_tmp = []
    label_tmp = []
    label_gt_tmp = []
    point_tmp = []
    ratio_tmp = []
    b_map_tmp = []
    field_length = 20

    index_seg = 0
    r1, r2 = [None, None]
    c1, c2 = [None, None]
    for i in range(len(pixel_index[0])):
        x = pixel_index[0][i]
        y = pixel_index[1][i]
        class_i = mask_label[x, y] - 1

        if (x < size / 2) or (x >= v_num) or (y < size / 2) or (y >= h_num) or (pointMark[x, y] == 1):
            continue

        # mark the center point
        pointMark[x, y] = 1
        r1, r2 = [x - size // 2, x + size // 2]
        c1, c2 = [y - size // 2, y + size // 2]
        img_seg = img[r1:r2, c1:c2]
        mask_seg = mask[r1:r2, c1:c2]
        mask_label_seg = mask_label[r1:r2, c1:c2]
        mask_label_gt_seg = mask_label_gt[r1:r2, c1:c2]

        temp = np.array(mask_label_seg)
        temp[np.where(mask_label_seg) == 1] = 100
        temp[np.where(mask_label_seg) == 2] = 200
        temp[np.where(mask_label_seg) == 3] = 150
        temp[np.where(mask_label_seg) == 4] = 255
        temp[np.where(mask_label_seg) == 0] = 0

        class_0 = np.where(mask_label_seg == 1)
        class_1 = np.where(mask_label_seg == 2)
        class_2 = np.where(mask_label_seg == 3)
        class_3 = np.where(mask_label_seg == 4)
        class_non = np.where(mask_label_seg == 0)

        ratio_0 = len(class_0[0]) / (size * size)
        ratio_1 = len(class_1[0]) / (size * size)
        ratio_2 = len(class_2[0]) / (size * size)
        ratio_3 = len(class_3[0]) / (size * size)
        ratio_non = len(class_non[0]) / (size * size)

        if ratio_non < 0.45:
            # print('ratio_non', ratio_non)
            if class_i == 0:
                if ratio_0 > 0.99:
                    label_tmp.append(0)
                else:
                    continue
            elif class_i == 1:
                if ratio_0 > 0.8 or ratio_1 < 0.03 or ratio_2 > 0.5 or ratio_3 > 0.3:
                    continue
                else:
                    label_tmp.append(1)
            elif class_i == 2:
                if ratio_1 > 0.02 or ratio_2 < 0.99 or ratio_3 > 0.009:
                    continue
                else:
                    label_tmp.append(2)

            elif class_i == 3:
                # print('ratio_non', ratio_non)
                # print(ratio_0, ratio_1, ratio_2, ratio_3)
                if ratio_1 > 0.3 or ratio_0 > 0.8 or ratio_2 > 0.9:
                    continue
                else:
                    # print('vein--------------')
                    label_tmp.append(3)
            else:
                print('outliers!!!, class', class_i)
                continue
        else:
            # print('non label')
            continue
            # label_tmp.append(4)

        # y = class_i * ((class_i == 0) * (ratio_0 > 0.99) + (class_i == 1) + (class_i == 2) * (ratio_2 > 0.99) + (
        #             class_i == 3) * (ratio_1 == 0))

        for j in range(-field_length, field_length + 1):
            for k in range(-field_length, field_length + 1):
                if mask_label[x + j, y + k] == mask_label[x, y]:
                    pointMark[x + j, y + k] = 1


        label_gt = get_true_label(x, y, mask_label_gt, mask_label_gt_seg, size)
        label_gt_tmp.append(label_gt)
        img_tmp.append(img_seg)
        mask_tmp.append(mask_seg)
        point_tmp.append([x, y])
        ratio_tmp.append([ratio_0, ratio_1, ratio_2, ratio_3, ratio_non])
        b_map_tmp.append(temp)

        index_seg = index_seg + 1

    img_tmp = np.array(img_tmp)
    mask_tmp = np.array(mask_tmp)
    label_tmp = np.array(label_tmp)
    point_tmp = np.array(point_tmp)
    ratio_tmp = np.array(ratio_tmp)
    b_map_tmp = np.array(b_map_tmp)

    print('patch numbers:', len(img_tmp), len(mask_tmp), len(label_tmp), len(point_tmp))

    return img_tmp.copy(), mask_tmp.copy(), label_tmp.copy(), label_gt_tmp.copy(), point_tmp.copy(), ratio_tmp.copy(), b_map_tmp.copy()


def coarse_vein(dataset, model, rows, cols):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cols,
        shuffle=False,
        num_workers=4)
    model.eval()

    # shift could be a var from outside, and '2' can be obtained through shift
    return_pred = []
    return_label = []

    coarse_pred = np.zeros((rows*2, cols*2))
    coarse_label = np.zeros((rows*2, cols*2))
    tmp = np.zeros((rows,cols))
    index_tuple = np.where(tmp==0)
    index = [index_tuple[0]*2, index_tuple[1]*2]
    shift = [[0,0],[0,1],[1,0],[1,1]]

    single_image_num = 0
    image_single = np.zeros((rows*cols))
    label_single = np.zeros((rows*cols))
    with torch.no_grad():
        for batch_idx, (inputs, labels, labels_rough, labels_fine, masks, labels_int) in enumerate(data_loader):
            # aprint('batch idx:', batch_idx)
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels_rough = labels_rough.cuda()
            labels_fine = labels_fine.cuda()

            y_batch, loss, loss_split = model.eval_on_batch(inputs, labels_rough, labels_fine)

            y_pre = torch.argmax(y_batch['pred'], dim=1).cpu().numpy()
            y_label = torch.argmax(y_batch['label'], dim=1).cpu().numpy()
            offset = (batch_idx%rows) * cols
            image_single[np.where(y_pre==1)[0]+offset] = 1
            image_single[np.where(y_pre==3)[0]+offset] = 1
            label_single[np.where(y_label == 1)[0] + offset] = 1
            label_single[np.where(y_label == 3)[0] + offset] = 1

            if (batch_idx+1)%rows==0:
                k = single_image_num%len(shift)
                print('sub image', k)
                coarse_pred[(index[0]+shift[k][0], index[1]+shift[k][1])] = image_single
                coarse_label[(index[0]+shift[k][0], index[1]+shift[k][1])] = label_single
                single_image_num += 1
                # reset
                image_single = np.zeros((rows * cols))
                label_single = np.zeros((rows * cols))
                if k == len(shift)-1:
                    print('image', single_image_num/len(shift), 'done')
                    return_pred.append(coarse_pred)
                    return_label.append(coarse_label)
                    # reset
                    coarse_pred = np.zeros((rows * 2, cols * 2))
                    coarse_label = np.zeros((rows * 2, cols * 2))

    return return_pred, return_label

def infer(input):
    kernel = [[1, 1, 1],
              [1, 10, 1],
              [1, 1, 1]]
    # expand batch size and channel dimensions
    kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
    raw_pred = torch.Tensor(input).unsqueeze(0).unsqueeze(0)

    result = F.conv2d(raw_pred, kernel, padding=1)
    result = result.squeeze().numpy()

    input[np.where(result<=10)] = 0
    return input

def postprocess(coarse_pred, coarse_label):
    # return: [numpy, numpy, ... , numpy]
    coarse_pred = list(map(infer, coarse_pred))
    coarse_label = list(map(infer, coarse_label))

    partial_resize = functools.partial(cv2.resize, dsize=(1080,1440), interpolation=cv2.INTER_NEAREST)
    resize_pred = list(map(partial_resize, coarse_pred))
    resize_label = list(map(partial_resize, coarse_label))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    partial_dilate = functools.partial(cv2.dilate, kernel=kernel)
    dilate_pred = list(map(partial_dilate, resize_pred))

    coarse_pred = dilate_pred
    coarse_label = resize_label

    return coarse_pred, coarse_label

def extract_each_layer(image, threshold):
    """
    This image processing funtion is designed for the OCT image post processing.
    It can remove the small regions and find the OCT layer boundary under the specified threshold.
    :param image:
    :param threshold:
    :return:
    """
    # convert the output to the binary image
    ret, binary = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)

    bool_binary = np.array(binary, bool)

    # remove the small object
    remove_binary = morphology.remove_small_objects(bool_binary, min_size=25000,
                                                                connectivity=2,
                                                                in_place=False)
    c = np.multiply(bool_binary, remove_binary)
    final_binary = np.zeros(shape=np.shape(binary))
    final_binary[c == True] = 1
    binary_image = cv2.filter2D(final_binary, -1, np.array([[-1], [1]]))
    layer_one = np.zeros(shape=[1, np.shape(binary_image)[1]])
    for i in range(np.shape(binary_image)[1]):
        location_point = np.where(binary_image[:, i] > 0)[0]
        # print(location_point)

        if len(location_point) == 1:
            layer_one[0, i] = location_point
        elif len(location_point) == 0:
            layer_one[0, i] = layer_one[0, i-1]

        else:
            layer_one[0, i] = location_point[0]

    return layer_one


if __name__ == '__main__':
    image_path = 'oct202.png'
    gt_path = 'oct202.png'
    image = cv2.imread(image_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('gt.png', gt)
    print(np.max(image), np.shape(image))
    print(np.max(gt), np.shape(gt))