from mindspore import nn, ops
import mindspore
import numpy as np
import mmcv

from resnet import ResNetFea, ResidualBlockUsing
from fpn import FeatPyramidNeck
from rpn import EmbRPN
from roi_align import SingleRoIExtractor
from stqi_head import DIIHead
from mask_head import DynamicMaskHead


def bbox2roi(bbox_list):
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        img_inds = mindspore.Tensor(np.ones((bboxes.shape[0], 1), np.float32) * img_id)
        rois = mindspore.ops.concat([img_inds, bboxes[:, :4]], axis=-1)
        rois_list.append(rois)
    rois = mindspore.ops.concat(rois_list, axis=0)
    return rois


class TeViT(nn.Cell):
    def __init__(self):
        super(TeViT, self).__init__()

        # backbone network
        self.backbone = ResNetFea(
            ResidualBlockUsing,
            [3, 4, 6, 3],
            [64, 256, 512, 1024],
            [256, 512, 1024, 2048],
            False)

        # feature pyramid network
        self.fpn = FeatPyramidNeck(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256)

        # proposal network
        self.rpn = EmbRPN(100, 256)

        self.num_stages = 6

        # bbox heads
        self.bbox_roi_extractors = nn.CellList([
            SingleRoIExtractor(7, 256, 2) for _ in range(self.num_stages)])
        self.bbox_heads = nn.CellList([
            DIIHead(num_classes=40) for _ in range(self.num_stages)])

        # mask heads
        self.mask_roi_extractor = SingleRoIExtractor(14, 256, 2)
        self.mask_head = DynamicMaskHead(num_classes=40)

        # bbox coder
        self.bbox_std = mindspore.Tensor([.5, .5, 1., 1.]).reshape(1, -1)

    def construct(self, imgs, img_metas):
        x = self.backbone(imgs)
        x = self.fpn(x)
        bboxes, queries, imgs_whwh = self.rpn(img_metas)

        for stage in range(self.num_stages):
            rois = bbox2roi(bboxes)
            bboxes, cls_scores, queries, attn_feats = self._bbox_forward(stage, x, rois, queries, img_metas, imgs_whwh)

        num_classes = cls_scores.shape[-1]
        cls_scores_mean = cls_scores.sigmoid().mean(0).reshape(-1)
        scores, topk_indices = mindspore.ops.TopK(sorted=False)(cls_scores_mean, 10)
        labels = topk_indices % num_classes
        bboxes = bboxes[:, topk_indices // num_classes]
        attn_feats = attn_feats[:, topk_indices // num_classes]

        rois = bbox2roi(bboxes)
        mask_feats = self.mask_roi_extractor(rois, x)
        masks = self.mask_head(mask_feats, attn_feats).reshape(-1, 10, num_classes, 28, 28)

        return self.post_process_results(bboxes.asnumpy(), labels.asnumpy(), scores.asnumpy(), masks.asnumpy(), img_metas)

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas, imgs_whwh):
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractors[stage]
        bbox_head = self.bbox_heads[stage]
        bbox_feats = bbox_roi_extractor(rois, x)

        cls_scores, bbox_preds, object_feats, attn_feats = bbox_head(bbox_feats, object_feats)
        bboxes = self.refine_bboxes(rois, cls_scores, bbox_preds, img_metas, imgs_whwh)
        return bboxes, cls_scores, object_feats, attn_feats

    def refine_bboxes(self, rois, labels, bbox_preds, img_metas, imgs_whwh):
        n, b, c = labels.shape

        labels = labels.reshape(n*b, c)
        bbox_preds = bbox_preds.reshape(n*b, 4)

        denorm_deltas = bbox_preds * self.bbox_std
        dxy = denorm_deltas[:, :2]
        dwh = denorm_deltas[:, 2:]

        pxy = ((rois[:, 1:3] + rois[:, 3:]) * 0.5)
        pwh = (rois[:, 3:] - rois[:, 1:3])

        dxy_wh = pwh * dxy

        max_ratio = np.abs(np.log(16 / 1000))
        dwh = mindspore.ops.clip_by_value(dwh, clip_value_min=-max_ratio, clip_value_max=max_ratio)

        gxy = pxy + dxy_wh
        gwh = pwh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = mindspore.ops.concat([x1y1, x2y2], axis=-1)

        bboxes[..., 0::2] = mindspore.ops.clip_by_value(bboxes[..., 0::2], clip_value_min=0, clip_value_max=img_metas[0]['img_shape'][1])
        bboxes[..., 1::2] = mindspore.ops.clip_by_value(bboxes[..., 1::2], clip_value_min=0, clip_value_max=img_metas[0]['img_shape'][0])

        return bboxes.reshape(n, b, 4)

    def post_process_results(self, bboxes, labels, scores, masks, img_metas):
        B, N = bboxes.shape[0], bboxes.shape[1]
        
        bboxes = bboxes.reshape(-1, bboxes.shape[2])
        masks = masks[:, range(N), labels]
        masks = masks.reshape(-1, *masks.shape[2:])
        
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor'].asnumpy()
        
        bboxes = (bboxes / scale_factor).astype(np.int32)
        im_masks = np.zeros((B*N, ori_shape[0], ori_shape[1]), dtype=np.uint8)
        bboxes_valid = (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3])
        h, w = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]

        for _ in range(B*N):
            if not bboxes_valid[_]:
                continue
            mask = mmcv.imresize(masks[_], (h[_], w[_]))
            mask = (mask > .5).astype(np.uint8)

            im_masks[_][bboxes[_, 1]:bboxes[_, 3], bboxes[_, 0]:bboxes[_, 2]] = mask

        return bboxes.reshape(B, N, -1), labels, scores, im_masks.reshape(B, N, *im_masks.shape[1:])
