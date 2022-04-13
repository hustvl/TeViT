import json
import os.path as osp
from argparse import ArgumentParser
from threading import Thread
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import collate, scatter
from pycocotools import mask as mask_utils
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.core.bbox import bbox_overlaps
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--json',
        default='data/youtubevis/annotations/valid.json',
        help='Path to VIS json file')
    parser.add_argument(
        '--root', default='data/youtubevis/valid/', help='Path to image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))

def main(args):
    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)
    cfg = model.cfg
    anno = json.load(open(args.json))
    test_pipeline = Compose(cfg.data.test.pipeline)

    results = []
    for video in tqdm(anno['videos']):
        imgs = video['file_names']
        vid_name = imgs[0].split('/')[0]

        datas, threads = [], []
        for img in imgs:
            data = dict(img_info=dict(filename=img), img_prefix=args.root)
            threads.append(Thread(target=load_datas, args=(data, test_pipeline, datas)))
            threads[-1].start()
        for thread in threads:
            thread.join()

        datas = sorted(datas, key=lambda x:x['img_metas'].data['filename'])

        datas = collate(datas, samples_per_gpu=len(imgs))
        datas['img_metas'] = datas['img_metas'].data
        datas['img'] = datas['img'].data
        datas = scatter(datas, [args.device])[0]

        with torch.no_grad():
            (det_bboxes, det_labels), segm_masks = model(
                return_loss=False,
                rescale=True,
                format=False,
                **datas)

        det_bboxes = torch.stack(det_bboxes)
        det_labels = torch.stack(det_labels)
        segm_masks = torch.stack(segm_masks)

        for inst_ind in range(det_bboxes.size(1)):
            objs = dict(
                video_id=video['id'],
                score=det_bboxes[:, inst_ind,
                                        -1].mean().item(),
                category_id=det_labels[0, inst_ind].item() +
                1,
                segmentations=[])
            for sub_ind in range(segm_masks.size(0)):
                m = segm_masks[
                    sub_ind, inst_ind,
                    ...].detach().cpu().numpy().astype(np.uint8)
                m_ = mask_utils.encode(
                    np.array(
                        m[:, :, np.newaxis],
                        order='F',
                        dtype='uint8'))[0]
                m_['counts'] = m_['counts'].decode()
                objs['segmentations'].append(m_)
            results.append(objs)

    # export results to json format and calculate mean Average-Precision
    json.dump(results, open('results.json', 'w'))
    ytvos = YTVOS(args.json)
    ytvos_dets = ytvos.loadRes('results.json')

    # if no annotation available, cocoapi returns -1 by default
    vid_ids = ytvos.getVidIds()
    for res_type in ['segm']:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
        ytvosEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()

if __name__ == '__main__':
    args = parse_args()
    main(args)
