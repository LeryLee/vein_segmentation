import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from .point_features import (
    mask_rcnn_loss,
    roi_mask_point_loss,
    mask_rcnn_inference
)
# from ..utils.loss import calculate_IoU_Dice
# from .scheduler import WarmupLinearSchedule, WarmupCosineSchedule

import cv2
import numpy as np  


class MyFrame():
    def __init__(self, net, loss, args, evalmode=False, pointmode=True):

        self.net = net(args.num_subdivision_points).cuda()
        #self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=args.learning_rate)

        self.loss = loss()
        self.old_lr = args.learning_rate
        self.mask_point_on = True
        self.lambda_dice_iou_loss = args.lambda_dice_iou_loss
        if not pointmode:
            self.mask_point_on = False
        if evalmode:
            # for i in self.net.modules():
            #     if isinstance(i, nn.BatchNorm2d):
            #         i.eval()
            self.net.eval()

    def train(self):
        # for i in self.net.modules():
        #     if isinstance(i, nn.BatchNorm2d):
        #         i.train()
        self.net.train()

    def eval(self):
        # for i in self.net.modules():
        #     if isinstance(i, nn.BatchNorm2d):
        #         i.eval()
        self.net.eval()
        
    def set_input(self, img_batch, mask_batch=None, b_map_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.b_map = b_map_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask

    def test(self):
        #
        # if mask_point on
        # return: loss.data, pred, result
        # else
        # return: loss.data, pred
        #
        self.forward()
        # pred = self.net.forward(self.img)
        if self.mask_point_on:
            self.net.mask_point_on = True
            result = self.net.forward(self.img)
            mask_coarse_logits = result['mask_coarse_logits']
            point_coords_wrt_image = result['point_coords_wrt_image']
            mask_point_logits = result['mask_point_logits']
            mask = self.mask


            loss_dict = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, mask, self.b_map)}
            loss_dict.update({
                    "loss_mask_point": roi_mask_point_loss(
                        mask_point_logits, mask, point_coords_wrt_image
                    )
                })

            loss = sum(loss_dict.values())
            pred = result['mask_logits'].sigmoid()
            point_indices = result['point_indices']

            return loss.data, pred, result
            
        else:
            self.net.mask_point_on = False
            result = self.net.forward(self.img)
            mask_coarse_logits = result['mask_coarse_logits']
            mask = self.mask

            loss_dict = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, mask, self.b_map)}
            
            loss = sum(loss_dict.values())
            pred = result['mask_coarse_logits'].sigmoid()

            return loss.data, pred

    # transfer img and mask to cuda
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            # self.mask = V(self.mask.cuda(), volatile=volatile)
            self.mask = self.mask.cuda()
        if self.b_map is not None:
            self.b_map = self.b_map.cuda()
        
    def optimize(self):
        #
        # return: loss.data, pred
        #
        self.forward()
        self.optimizer.zero_grad()

        if self.mask_point_on:
            self.net.mask_point_on = True
            result = self.net.forward(self.img)
            mask_coarse_logits = result['mask_coarse_logits']
            point_coords_wrt_image = result['point_coords_wrt_image']
            mask_point_logits = result['mask_point_logits']
            mask = self.mask


            loss_dict = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, mask, self.b_map)}
            loss_dict.update({
                    "loss_mask_point": roi_mask_point_loss(
                        mask_point_logits, mask, point_coords_wrt_image
                    )
                })
            
            pred = result['mask_logits'].sigmoid()

        else:
            self.net.mask_point_on = False
            result = self.net.forward(self.img)
            mask_coarse_logits = result['mask_coarse_logits']
            mask = self.mask

            loss_dict = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, mask, self.b_map)}

            pred = result['mask_coarse_logits'].sigmoid()
        
        _, loss_dice_iou = self.loss(mask, pred)
        loss_dict.update({ "loss_dice_iou": self.lambda_dice_iou_loss * loss_dice_iou })

        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

        return loss.data, pred
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        self.old_lr = new_lr
