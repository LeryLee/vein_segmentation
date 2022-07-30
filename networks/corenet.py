import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import glog as log 

from functools import partial

from .shape_spec import ShapeSpec
from .wrappers import cat
from .point_features import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
    calculate_uncertainty,
    roi_mask_point_loss,
)


nonlinearity = partial(F.relu, inplace=True)

__all__ = [
           "SPPblock_original",
           "DACblock_original",
           "CoRE_Net_Encoder",
           "CoRE_Net_Decoder",
           "StandardPointHead",
           "CoRE_Net"
           ]


class DACblock_original(nn.Module):
    def __init__(self, channel):
        super(DACblock_original, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


# class PSPModule(nn.Module):
#     def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
#         super().__init__()
#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
#         self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
#         self.relu = nn.ReLU()

#     def _make_stage(self, features, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
#         return nn.Sequential(prior, conv)

#     def forward(self, feats):
#         h, w = feats.size(2), feats.size(3)
#         priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
#         bottle = self.bottleneck(torch.cat(priors, 1))
#         return self.relu(bottle)


class SPPblock_original(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock_original, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CoRE_Net_Encoder(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CoRE_Net_Encoder, self).__init__()

        filters = [64, 128, 256, 516]

        self._out_features = ['e1', 'e2', 'e3', 'e4']
        self._out_feature_channels = {name: filters[idx]
                                      for idx, name  in enumerate(self._out_features)} # {64,128,256,516}
        self._out_feature_strides = {name: 2**(idx+2)
                                     for idx, name in enumerate(self._out_features)} # {4,8,16,32}

        resnet = models.resnet34(pretrained=False)
        # weight = .clone()
        self.firstconv = resnet.conv1
        #self.firstconv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.firstconv.weight[:,:3] = resnet.conv1.weight.clone()
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock_original(512)
        self.spp = SPPblock_original(512)

    def forward(self, x):
        outputs = {}
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        outputs['e1'] = e1
        e2 = self.encoder2(e1)
        outputs['e2'] = e2
        e3 = self.encoder3(e2)
        outputs['e3'] = e3
        e4 = self.encoder4(e3)

        # # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        outputs['e4'] = e4
        # Decoder
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

class CoRE_Net_Decoder(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CoRE_Net_Decoder, self).__init__()

        filters = [64, 128, 256, 516]
        self._out_features = ['d1', 'd2', 'd3', 'd4', 'd5']
        self._out_feature_channels = {name: filters[0] if idx==0 else filters[idx-1]
                                      for idx, name in enumerate(self._out_features)} # {64,64,128,256,516}
        self._out_feature_strides = {name: 2 ** (idx + 1)
                                     for idx, name in enumerate(self._out_features)}  # {2,4,8,16,32}

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2]+filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1]+filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0]+filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, features_dict_former):
        outputs = {}
        # Encoder
        e1 = features_dict_former['e1']
        e2 = features_dict_former['e2']
        e3 = features_dict_former['e3']

        # Center
        e4 = features_dict_former['e4']

        outputs['d5'] = e4
        # Decoder
        d4 = self.decoder4(e4)
        outputs['d4'] = d4
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        outputs['d3'] = d3
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        outputs['d2'] = d2
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        outputs['d1'] = d1

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        outputs['logits'] = out
        # F.sigmoid(out)

        return outputs
        # return out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self):
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
        fc_dim                      = [512,256,256,64]
        num_fc                      = 4
        cls_agnostic_mask           = False
        self.coarse_pred_each_layer = True
        input_channels              = [1028,512,256,256,64]
        # fmt: on

        fc_dim_in = input_channels[0] + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim[k], kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = input_channels[k+1]
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        fc_dim_in = input_channels[-1]
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
        for k, layer in enumerate(self.fc_layers):
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)


class CoRE_Net(nn.Module):
    def __init__(self, num_subdivision_points, num_classes=1, num_channels=3):
        super(CoRE_Net, self).__init__()

        self.backbone = CoRE_Net_Encoder()
        self.mask_coarse_head = CoRE_Net_Decoder()
        self.mask_point_head = StandardPointHead()

        self._init_mask_head(self.backbone.output_shape())
        self._init_point_head(num_subdivision_points, self.mask_coarse_head.output_shape())

        filters = [64, 128, 256, 516]
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def _init_mask_head(self, input_shape):

        self.mask_on                 = True 
        if not self.mask_on:
            return
        self.mask_coarse_in_features = [k for k, v in input_shape.items()] 
        self.mask_coarse_side_size   = 14 


    def _init_point_head(self, num_subdivision_points, input_shape):

        self.mask_point_on                      = True 
        if not self.mask_point_on:
            return
        # 
        self.mask_point_in_features             = [k for k, v in input_shape.items()]
        self.mask_point_train_num_points        = num_subdivision_points 
        self.mask_point_oversample_ratio        = 3 
        self.mask_point_importance_sample_ratio = 0.75
        # 
        self.mask_point_subdivision_steps       = 5 
        self.mask_point_subdivision_num_points  = num_subdivision_points 
        self._feature_scales                    = {k: 1.0 / v.stride for k, v in input_shape.items()}

        in_channels = np.sum([input_shape[f].channels for f in self.mask_point_in_features])


    def forward(self, batch_inputs):
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

        if self.training:
            former_features = self.backbone(batch_inputs)
            features = self._forward_mask_coarse(former_features)
            mask_coarse_logits = features['logits']

            if self.mask_point_on:
                result = self._forward_mask_point(features, mask_coarse_logits)
                point_logits = result['point_logits']
                point_coords_wrt_image = result['point_coords_wrt_image']
                mask_logits = result['mask_logits']
                point_indices = result['point_indices']
                point_coords = result['point_coords']

                return {'mask_coarse_logits': mask_coarse_logits,
                        'point_coords_wrt_image': point_coords_wrt_image,
                        'point_coords': point_coords,
                        'point_indices': point_indices,
                        'mask_point_logits': point_logits,
                        'mask_logits': mask_logits
                        }
            else:
                return {'mask_coarse_logits': mask_coarse_logits}
        else:
            
            former_features = self.backbone(batch_inputs)
            features = self._forward_mask_coarse(former_features)
            mask_coarse_logits = features['logits']

            if self.mask_point_on:
                result = self._forward_mask_point(features, mask_coarse_logits)
                point_logits = result['point_logits']
                point_coords_wrt_image = result['point_coords_wrt_image']
                mask_logits = result['mask_logits']
                point_indices = result['point_indices']
                point_coords = result['point_coords']
                
                return {'mask_coarse_logits': mask_coarse_logits,
                        'point_coords_wrt_image': point_coords_wrt_image,
                        'point_coords': point_coords,
                        'point_indices': point_indices,
                        'mask_point_logits': point_logits,
                        'mask_logits': mask_logits
                        }
            else:
                return {'mask_coarse_logits': mask_coarse_logits}


    def _forward_mask(self, features, instances):

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

    def _forward_mask_coarse(self, former_features):
        """
        Forward logic of the coarse mask head.
        """

        return self.mask_coarse_head(former_features)

    def _forward_mask_point(self, features, mask_coarse_logits):
        """
        Forward logic of the mask point head.
        """
        if not self.mask_point_on: 
            return {} if self.training else mask_coarse_logits

        mask_features_list = [features[k] for k in self.mask_point_in_features] # ('d1', 'd2', 'd3', 'd4', 'd5')
        features_scales = [self._feature_scales[k] for k in self.mask_point_in_features] # [1.0/4]

        if self.training:
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits),
                    self.mask_point_train_num_points, # 14*14
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, features['logits'], point_coords
            )
            coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)

            # put mask point predictions to the right places on the upsampled grid.
            R, C, H, W = mask_coarse_logits.shape
            h_step = 1.0 / float(H)
            w_step = 1.0 / float(W)
            
            point_indices = ((point_coords[:, :, 1] - (h_step / 2.0)) * h_step + \
                            (point_coords[:, :, 0] - (w_step / 2.0)) * w_step).to(torch.int64)
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)

            mask_logits = torch.scatter(mask_coarse_logits.reshape(R, C, H * W),
                                        2, point_indices, point_logits).view(R, C, H, W)
            return {'point_coords_wrt_image': point_coords_wrt_image,
                    'point_coords': point_coords,
                    'point_indices': point_indices,
                    'point_logits': point_logits,
                    'mask_logits': mask_logits
                    }
        else:

            mask_logits = mask_coarse_logits.clone()

            # If `mask_point_subdivision_num_points` is larger or equal to the
            # resolution of the next step, then we can skip this step
            H, W = mask_logits.shape[-2:]

            uncertainty_map = calculate_uncertainty(mask_logits)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map, self.mask_point_subdivision_num_points
            )
            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, features['logits'], point_coords
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
            return {'point_coords_wrt_image': point_coords_wrt_image,
                    'point_coords': point_coords,
                    'point_indices': point_indices,
                    'point_logits': point_logits,
                    'mask_logits': mask_logits
                    }



class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

