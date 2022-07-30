from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import os 
import os.path as osp
import glog as log

import torch
# import torch.utils.data as data
import numpy as np
import cv2
import h5py
import pickle
from scipy.ndimage import convolve
import pdb

from networks.point_features import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
    calculate_uncertainty,
    roi_mask_point_loss,
)
from utils.gvf import (
    gradient_vector_flow,
    edge_map,
    gradient_field,
    add_border
)

def element_wise_minimum(np1, np2):
    result = np.minimum(np1, np2)
    return result

def element_wise_maximum(np1, np2):
    result = np.maximum(np1, np2)
    return result

def element_wise_add(np1, np2):
    result = np1 + np2
    return result

def element_wise_and(np1, np2):
    result = np.bitwise_and(np1, np2)
    return result

def element_wise_avg(np1, np2):
    result = (np1 + np2)//2
    return result

def cal_grad(img):
    edge = edge_map(img, sigma=2)
    # calc GVF
    fx, fy = gradient_field(edge)
    gx, gy = gradient_vector_flow(fx, fy, mu=1.0)

    return fx,fy,gx,gy


def infer_break(fx, fy, tau=4):
    fa = np.sqrt(fx*fx + fy*fy)
    kernel = np.array([
        [1, 1, 1],
        [1, -10, 1],
        [1, 1, 1]
    ])
    res = np.zeros(fa.shape)
    res[np.where(fa > fa.max()*0.6)] = 1
    index_break = np.where(res > tau)

    return index_break


def infer_branching(gx, gy):
    fxx, fxy = gradient_field(gx)
    fyx, fyy = gradient_field(gy)
    div_img = fxx + fyy
    index_branching = np.where(div_img <= div_img.min()*0.78)

    return index_branching


def point_refiner(index_infer, mask_coarse_logits, mask_features_list, features_scales, features, solver, point_indices=None):

    R, C, H, W = mask_coarse_logits.shape
    if point_indices == None:
        point_indices = torch.LongTensor(index_infer[0] * W + index_infer[1]).cuda()
    point_coords[:, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step

    fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
        mask_features_list, features_scales, features['logits'], point_coords
    )
    coarse_features = point_sample(
        mask_coarse_logits, point_coords, align_corners=False
    )
    point_logits = solver.net.mask_point_head(fine_grained_features, coarse_features)

    return point_logits

def filling_with_points(mask_coarse_logits, point_logits, index_infer, point_indices=None):
    # put mask point predictions to the right places on the upsampled grid.
    mask_logits = mask_coarse_logits.clone()
    R, C, H, W = mask_logits.shape
    if point_indices == None:
        point_indices = torch.LongTensor(index_infer[0] * W + index_infer[1])
    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    mask_logits = (
        mask_logits.reshape(R, C, H * W)
        .scatter_(2, point_indices, point_logits.cpu())
        .view(R, C, H, W)
    )

    return mask_logits


def CMM(mask_logits):

    uncertainty_map = calculate_uncertainty(mask_logits)
    point_indices_uncertain, point_coords = get_uncertain_point_coords_on_grid(
        uncertainty_map, args.num_subdivision_points
    )

    return point_indices_uncertain


def PCM(pred):
    bb_map = np.zeros(b_map.shape)
    bb_map[np.where(b_map > 0.7)] = 1
    bb_map[np.where(b_map < 0.1)] = 1
    img_coarse = pred * bb_map

    fx,fy,gx,gy = cal_grad(img_coarse)
    index_break = infer_break(fx, fy)
    index_branching = infer_branching(gx, gy)

    return index_break, index_branching


def infer_point(args, dataset, solver, threshold, outer_epoch, mode):
    '''infer the point labels and take place the corresponding positions.

    :param dataset: trainset when training, testset when final testing.
    :param solver: CFNet model
    :param threshold: low and high threshold for confidence inference
    :param save_folder: h5 output folder
    :return: pred_results: [0,255] images, for the new masks and for visualization
             b_map_results: {0,1} binary confidence maps
             logit_results: [0,1] float matrix, the original output after point replacement
    '''
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8)


    image_list = dataset.image_names

    pred_results = list()
    b_map_results = list()
    logit_results = list()

    solver.eval()
    with torch.no_grad():
        for batch_idx, (img, mask, _) in enumerate(data_loader):
            img = img.cuda()

            former_features = solver.net.backbone(img)
            features = solver.net._forward_mask_coarse(former_features)
            mask_coarse_logits = features['logits']
            pred = mask_coarse_logits.sigmoid()

            mask_features_list = [features[k] for k in solver.net.mask_point_in_features] # ('d1', 'd2', 'd3', 'd4', 'd5') [[1,256,184,336]]
            features_scales = [solver.net._feature_scales[k] for k in solver.net.mask_point_in_features]

            R, C, H, W = pred.shape
            # normlization
            pred_1 = pred.reshape(R*C, -1) 
            min_v = pred_1.min(dim=1)[0].reshape(R, C, 1, 1)
            max_v = pred_1.max(dim=1)[0].reshape(R, C, 1, 1)
            pred = (pred - min_v) / (max_v - min_v)


            pred = pred.squeeze().cpu()
            show_pred = pred * 255.
            show_pred = show_pred.int().numpy()
            show_pred = np.array(show_pred, np.uint8)

            _, show_pred = cv2.threshold(show_pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pred_results.append(show_pred)

            logit = pred.squeeze().cpu()
            logit_results.append(logit.numpy())

            mask_logits = mask_coarse_logits.clone()
            if args.confidence_on:
                b_map = torch.zeros(pred.size())

                pred_map = pred.cpu()
                b_map[pred_map >= threshold['high']] = 1
                b_map[pred_map <= threshold['low']] = 1

                point_indices_uncertain = CMM(mask_logits)
                point_logits = point_refiner(None, mask_coarse_logits, mask_features_list, features_scales, features, solver, point_indices_uncertain)
                mask_logits = filling_with_points(mask_logits, point_logits, None, point_indices_uncertain)

                fill_ones = torch.ones(point_indices_uncertain.size())
                b_map = (
                    b_map.reshape(R, C, H * W)
                        .scatter_(2, point_indices_uncertain.cpu(), fill_ones)
                        .view(R, C, H, W)
                )
            else:
                b_map = torch.ones(pred.size())                
                
            if args.point_correction_on:
                index_break, index_branching = PCM(pred)
                point_logits = point_refiner(index_break, mask_coarse_logits, mask_features_list, features_scales, features, solver)
                mask_logits = filling_with_points(mask_logits, point_logits, index_break)
                b_map[index_break] = 1

                point_logits = point_refiner(index_branching, mask_coarse_logits, mask_features_list, features_scales, features, solver)
                mask_logits = filling_with_points(mask_logits, point_logits, index_branching)
                b_map[index_branching] = 1             

            b_map = b_map.squeeze().numpy()
            b_map_results.append(b_map)


            log.info('image {} {} done'.format(batch_idx, image_list[batch_idx]))

    return pred_results, b_map_results, logit_results


def infer_labels(args, trainset, testset, solver, outer_epoch):
    # infer the labels for dataset
    # before self-training

    th_split = args.if_threshold.split(',')
    threshold = {'high': float(th_split[0]), 'low': float(th_split[1])}
    pred_results, b_map_results, logit_results = infer_point(args, trainset, solver, threshold, outer_epoch, mode='train')

    # pred_results_test, b_map_results_test, logit_results_test = infer_point(args, testset, solver, threshold, outer_epoch, mode='test')

    trainset.masks = pred_results
    trainset.b_maps = b_map_results
    trainset.is_random = True

    return trainset
