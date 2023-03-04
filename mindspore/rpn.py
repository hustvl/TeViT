import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F


def bbox_cxcywh_to_xyxy(bbox):
    split = mindspore.ops.Split(1, 4)
    cx, cy, w, h = split(bbox)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return mindspore.ops.concat(bbox_new, axis=-1)


class EmbRPN(nn.Cell):

    def __init__(self,
                 num_proposals=100,
                 proposal_feature_channel=256):
        super(EmbRPN, self).__init__()
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel

        init_proposal_bboxes = np.zeros((num_proposals, 4))
        init_proposal_features = np.random.randn(num_proposals, proposal_feature_channel)

        init_proposal_bboxes[:, :2] = .5
        init_proposal_bboxes[:, 2:] = 1.
        
        self.init_proposal_bboxes = mindspore.Parameter(init_proposal_bboxes.astype(np.float32))
        self.init_proposal_features = mindspore.Parameter(init_proposal_features.astype(np.float32))

    def construct(self, img_metas):
        proposals = self.init_proposal_bboxes.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)

        num_imgs = len(img_metas)
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(mindspore.Tensor(np.array([[w, h, w, h]]).astype(np.int32)))
        imgs_whwh = mindspore.ops.concat(imgs_whwh, axis=0)
        imgs_whwh = imgs_whwh[:, None, :]

        proposals = proposals * imgs_whwh

        init_proposal_features = self.init_proposal_features.clone()
        init_proposal_features = mindspore.numpy.repeat(init_proposal_features[None], num_imgs, axis=0)
        return proposals, init_proposal_features, imgs_whwh
