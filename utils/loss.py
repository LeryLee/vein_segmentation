import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2
from terminaltables import AsciiTable
import pdb

def compute_iou(y_pred, y_true):
    ## input: y_pred, y_true, NxCxHxW, C=1

    smooth = 0.0 # may change
    N = len(y_pred)
    y_pred = y_pred.squeeze().cpu().numpy() # NxHxW
    y_true = y_true.squeeze().cpu().numpy() # NxHxW
    
    # cv2 adaptive threshold 
    y_pred = np.array(y_pred * 255, np.uint8)
    y_pred_bi = list()
    if N > 1:
        for i in range(N):
            _, _y_pred = cv2.threshold(y_pred[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            y_pred_bi.append(_y_pred)
            # y_pred_bi.append(y_pred[i])
    else:
        _, _y_pred = cv2.threshold(y_pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y_pred_bi.append(_y_pred)
        # y_pred_bi.append(y_pred[i])
        y_true = np.expand_dims(y_true, axis=0)
    y_pred = np.array(y_pred_bi) / 255
    # y_pred = y_pred / 255

    i = y_true.sum(1).sum(1)
    j = y_pred.sum(1).sum(1)
    intersection = (y_true * y_pred).sum(1).sum(1)

    # dice_score = (2. * intersection + smooth) / (i + j + smooth)
    iou_score = (intersection + smooth) / (i + j - intersection + smooth)
    # iou_score = iou_score.sum()/N
    # will divide by N in the main()
    iou_score = iou_score.sum()

    return iou_score

def cal_per_class_loss(loss, label):
    num_classes = label.size(1)
    label = label.argmax(dim=1)
    loss_split = torch.zeros((num_classes))

    for i in range(num_classes):
        index = (label==i).nonzero().squeeze()
        if index.numel() == 0:
            loss_split[i] = 0
        else:
            loss_split[i] = torch.mean(loss[index])

    return loss_split

def contrastive_loss(distance, targets, margin=2.0):
    # contrastive loss
    # euclidean_distance
    # targets: [0, 1] 1: same 0: different
    loss = targets * torch.pow(distance, 2) + \
              (1 - targets) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    # print('loss_batch: ',loss_batch)

    return loss

def weigthed_cross_entropy_onehot(pred, soft_targets, eps=1e-10, weight=None):
    # weight's size is (1,4), for example [[1,1,1,1]]
    logsoftmax = nn.LogSoftmax(dim=1)
    logits = F.softmax(pred, dim=1)
    if weight is not None:
        batch_size = pred.size()[0]
        weight = weight.repeat(batch_size, 1)
        loss = torch.sum(- weight * soft_targets * logsoftmax(pred), 1)
        #loss = torch.sum(- weight * soft_targets * torch.log(logits + eps), 1)
    else:
        loss = torch.sum(- soft_targets * logsoftmax(pred), 1)
        #loss = torch.sum(- soft_targets * torch.log(logits + eps), 1)

    return loss

class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a

class MulticlassCELoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassCELoss, self).__init__()

    def forward(self, soft_targets, pred, weight=None):

        C = soft_targets.size(1)

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        # size = soft_targets.size()
        pred = pred.permute(0, 2, 3, 1).reshape(-1, C)
        soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, C)

        totalLoss = weigthed_cross_entropy_onehot(pred, soft_targets, weight=None)

        totalLoss = torch.mean(totalLoss)
        return totalLoss

class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b



def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.weight_bce_loss = nn.BCELoss(reduction='none')

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        score_iou = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean(), score_iou.mean()

    def soft_dice_loss(self, y_true, y_pred):
        score_dice, score_iou = self.soft_dice_coeff(y_true, y_pred)
        loss_dice = 1 - score_dice
        loss_iou = 1 - score_iou
        return 0.5*loss_dice + 0.5*loss_iou

    def __call__(self, y_true, y_pred, weighted=False):
        if weighted is True:
            a = self.weight_bce_loss(y_pred, y_true)
        else:
            a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a, b


import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target, input):
        target1 = torch.squeeze(target, dim=1)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target2 = target1.view(-1,1).long()

        logpt = F.log_softmax(input, dim=1)
        # print(logpt.size())
        # print(target2.size())
        logpt = logpt.gather(1,target2)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args: 
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

#     if isinstance(label, str):
#         label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

#     mask = (label != ignore_index)
#     pred_label = pred_label[mask]
#     label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


def evaluate(results,
             gt_seg_maps,
             metric='mIoU',
             logger=None,
             efficient_test=False,
             is_printTable=True,
             **kwargs):
    """Evaluate the dataset.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
            'mDice' are supported.
        logger (logging.Logger | None | str): Logger used for printing
            related information during evaluation. Default: None.

    Returns:
        dict[str, float]: Default metrics.
    """

    CLASSES = ('background', 'vein')
    if isinstance(metric, str):
        metric = [metric]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metric).issubset(set(allowed_metrics)):
        raise KeyError('metric {} is not supported'.format(metric))
    eval_results = {}
    # gt_seg_maps = self.get_gt_seg_maps(efficient_test)
    if CLASSES is None:
        num_classes = len(
            reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
    else:
        num_classes = len(CLASSES)
    ret_metrics = eval_metrics(
        results,
        gt_seg_maps,
        num_classes,
        metric
        # label_map=self.label_map,
        # reduce_zero_label=self.reduce_zero_label
        )
    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    if CLASSES is None:
        class_names = tuple(range(num_classes))
    else:
        class_names = CLASSES
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                            ['m' + head
                            for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                [ret_metrics_mean[1]] +
                                [ret_metrics_mean[0]])
    
    if is_printTable:
        logger.info('per class results:')
        # print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        # print_log('\n' + table.table, logger=logger)
        # print_log('Summary:', logger)
        logger.info('\n' + table.table)
        logger.info('Summary:')
        table = AsciiTable(summary_table_data)
        # print_log('\n' + table.table, logger=logger)
        logger.info('\n' + table.table)

    for i in range(1, len(summary_table_data[0])):
        eval_results[summary_table_data[0]
                        [i]] = summary_table_data[1][i] / 100.0
    # if mmcv.is_list_of(results, str):
    #     for file_name in results:
    #         os.remove(file_name)
    return eval_results

def calculate_IoU_Dice(results,
                       gt_seg_maps,
                       logger=None,
                       is_printTable=False):
    
    # results: list[torch.Tensor]
    # gt_seg_maps: list[torch.Tensor]
    # binary 
    results_bi = list()
    for i in range(len(results)):
        pred = np.array(results[i] * 255, np.uint8)
        _, _y_pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _y_pred = torch.tensor(_y_pred, dtype=torch.float)
        results_bi.append(_y_pred/255)

    IoU_results = evaluate(results_bi, 
                            gt_seg_maps, 
                            metric='mIoU', 
                            logger=logger,
                            is_printTable=is_printTable)
    Dice_results = evaluate(results_bi, 
                            gt_seg_maps, 
                            metric='mDice', 
                            logger=logger,
                            is_printTable=is_printTable)

    return IoU_results, Dice_results