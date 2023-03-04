import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.common.tensor import Tensor


class ROIAlign(nn.Cell):
    def __init__(self,
                 out_size_h,
                 out_size_w,
                 spatial_scale,
                 sample_num=0,
                 roi_align_mode=1):
        super(ROIAlign, self).__init__()

        self.out_size = (out_size_h, out_size_w)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.align_op = mindspore.ops.ROIAlign(self.out_size[0], self.out_size[1],
                                   self.spatial_scale, self.sample_num, roi_align_mode)

    def construct(self, features, rois):
        return self.align_op(features, rois)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        return format_str


class SingleRoIExtractor(nn.Cell):

    def __init__(self,
                 out_size=7,
                 out_channels=256,
                 sample_num=2,
                 featmap_strides=[4, 8, 16, 32],
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.sample_num = sample_num
        self.finest_scale = finest_scale

        self.roi_layers = nn.CellList(self.build_roi_layers())

    def build_roi_layers(self):
        roi_layers = []
        for s in self.featmap_strides:
            layer_cls = ROIAlign(self.out_size, self.out_size,
                                 spatial_scale=1 / s,
                                 sample_num=self.sample_num,
                                 roi_align_mode=0)
            roi_layers.append(layer_cls)
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        scale = mindspore.numpy.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = mindspore.numpy.floor(mindspore.numpy.log2(scale / self.finest_scale + 1e-6))
        target_lvls = mindspore.ops.clip_by_value(target_lvls, clip_value_min=0, clip_value_max=num_levels - 1)
        target_lvls = mindspore.ops.cast(target_lvls, mindspore.int64)
        return target_lvls

    def construct(self, rois, feats):
        num_levels = len(feats)
        roi_feats = mindspore.ops.zeros((rois.shape[0], self.out_channels, self.out_size, self.out_size), mindspore.float32)
        target_lvls = self.map_roi_levels(rois, num_levels)
        for i in range(num_levels):
            mask = target_lvls == i
            inds = mindspore.ops.nonzero(mask)
            if inds.size > 0:
                inds = mindspore.ops.squeeze(inds, axis=1)
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
        return roi_feats
