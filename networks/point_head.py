import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .wrappers import cat
from .shape_spec import ShapeSpec
from detectron2.structures import BitMasks
from detectron2.utils.events import get_event_storage

from .point_features import point_sample

"""
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def roi_mask_point_loss(mask_logits, instances, points_coord):
    """
    Compute the point-based loss for instance segmentation mask predictions.

    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the `mask_logits`. So, i_th
            elememt of the list contains R_i objects and R_1 + ... + R_N is equal to R.
            The ground-truth labels (class, box, mask, ...) associated with each instance are stored
            in fields.
        points_coords (Tensor): A tensor of shape (R, P, 2), where R is the total number of
            predicted masks and P is the number of points for each mask. The coordinates are in
            the image pixel coordinate space, i.e. [0, H] x [0, W].
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    assert len(instances) == 0 or isinstance(
        instances[0].gt_masks, BitMasks
    ), "Point head works with GT in 'bitmask' format only. Set INPUT.MASK_FORMAT to 'bitmask'."
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)

        gt_classes = []
        gt_mask_logits = []
        idx = 0
        for instances_per_image in instances:
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_bit_masks = instances_per_image.gt_masks.tensor
            h, w = instances_per_image.gt_masks.image_size
            scale = torch.tensor([w, h], dtype=torch.float, device=gt_bit_masks.device)
            points_coord_grid_sample_format = (
                points_coord[idx : idx + len(instances_per_image)] / scale
            )
            idx += len(instances_per_image)
            gt_mask_logits.append(
                point_sample(
                    gt_bit_masks.to(torch.float32).unsqueeze(1),
                    points_coord_grid_sample_format,
                    align_corners=False,
                ).squeeze(1)
            )
        gt_mask_logits = cat(gt_mask_logits)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if gt_mask_logits.numel() == 0:
        return mask_logits.sum() * 0

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.to(dtype=torch.uint8)
    mask_accuracy = mask_accurate.nonzero().size(0) / mask_accurate.numel()
    get_event_storage().put_scalar("point_rend/accuracy", mask_accuracy)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), reduction="mean"
    )
    return point_loss


class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        num_classes                 = 1
        fc_dim                      = cfg.MODEL.POINT_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC
        cls_agnostic_mask           = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        input_channels              = input_shape.channels
        # fmt: on

        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)



def calculate_uncertainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -torch.abs(gt_class_logits)


class PointRendROIHeads(StandardROIHeads):
    """
    The RoI heads class for PointRend instance segmentation models.

    In this class we redefine the mask head of `StandardROIHeads` leaving all other heads intact.
    To avoid namespace conflict with other heads we use names starting from `mask_` for all
    variables that correspond to the mask head in the class's namespace.
    """

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on                 = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        self.mask_coarse_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES # 'p2'
        self.mask_coarse_side_size   = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION # 14
        self._feature_scales          = {k: 1.0 / v.stride for k, v in input_shape.items()} # {'p2':1.0/4, 'p3':1.0/8, 'p4':1.0/16, 'p5':1.0/32, 'p6':1.0/64}
        # fmt: on

        in_channels = np.sum([input_shape[f].channels for f in self.mask_coarse_in_features]) # 256
        self.mask_coarse_head = build_mask_head(
            cfg,
            ShapeSpec(
                channels=in_channels, # 256
                width=self.mask_coarse_side_size, # 14
                height=self.mask_coarse_side_size, # 14
            ),
        )
        self._init_point_head(cfg, input_shape)

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on                      = cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON # True
        if not self.mask_point_on:
            return
        assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        self.mask_point_in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES # 'p2'
        self.mask_point_train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS # 196 = 14*14
        self.mask_point_oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO # 3
        self.mask_point_importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO # 0.75
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = np.sum([input_shape[f].channels for f in self.mask_point_in_features])
        self.mask_point_head = build_point_head(
            cfg, ShapeSpec(channels=in_channels, width=1, height=1)
        )

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_coarse_logits = self._forward_mask_coarse(features, proposal_boxes) # features = self.backbone(images.tensor)

            losses = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, proposals)}
            losses.update(self._forward_mask_point(features, mask_coarse_logits, proposals))
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_coarse_logits = self._forward_mask_coarse(features, pred_boxes)

            mask_logits = self._forward_mask_point(features, mask_coarse_logits, instances)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_mask_coarse(self, features, boxes):
        """
        Forward logic of the coarse mask head.
        """
        # point_coords (R, 14*14, 2)
        point_coords = generate_regular_grid_point_coords(
            np.sum(len(x) for x in boxes), self.mask_coarse_side_size, boxes[0].device
        )
        # mask_coarse_features_list [[1, 256, 184, 336]]
        mask_coarse_features_list = [features[k] for k in self.mask_coarse_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_coarse_in_features] # [1.0/4]
        # For regular grids of points, this function is equivalent to `len(features_list)' calls
        # of `ROIAlign` (with `SAMPLING_RATIO=2`), and concat the results.
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list, features_scales, boxes, point_coords
        ) # (R, 256, 14*14)
        return self.mask_coarse_head(mask_features)

    def _forward_mask_point(self, features, mask_coarse_logits, instances):
        """
        Forward logic of the mask point head.
        """
        if not self.mask_point_on:
            return {} if self.training else mask_coarse_logits

        mask_features_list = [features[k] for k in self.mask_point_in_features] # [[1,256,184,336]]
        features_scales = [self._feature_scales[k] for k in self.mask_point_in_features] # [1.0/4]

        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            gt_classes = cat([x.gt_classes for x in instances])
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits, gt_classes),
                    self.mask_point_train_num_points, # 14*14
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, proposal_boxes, point_coords
            )
            coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)
            return {
                "loss_mask_point": roi_mask_point_loss(
                    point_logits, instances, point_coords_wrt_image
                )
            }
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_classes = cat([x.pred_classes for x in instances])
            # The subdivision code will fail with the empty list of boxes
            if len(pred_classes) == 0:
                return mask_coarse_logits

            mask_logits = mask_coarse_logits.clone()
            for subdivions_step in range(self.mask_point_subdivision_steps):
                mask_logits = interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                # If `mask_point_subdivision_num_points` is larger or equal to the
                # resolution of the next step, then we can skip this step
                H, W = mask_logits.shape[-2:]
                if (
                    self.mask_point_subdivision_num_points >= 4 * H * W
                    and subdivions_step < self.mask_point_subdivision_steps - 1
                ):
                    continue
                uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.mask_point_subdivision_num_points
                )
                fine_grained_features, _ = point_sample_fine_grained_features(
                    mask_features_list, features_scales, pred_boxes, point_coords
                )
                coarse_features = point_sample(
                    mask_coarse_logits, point_coords, align_corners=False
                )
                point_logits = self.mask_point_head(fine_grained_features, coarse_features)

                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
            return mask_logits
