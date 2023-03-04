import json
import os.path as osp
from argparse import ArgumentParser
from threading import Thread
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
from mmdet.datasets.pipelines import Compose

from tevit import TeViT
import mindspore

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='Path to VIS json file')
    parser.add_argument('--root', type=str, required=True, help='Path to image file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    return args

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))

def main(args):
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='Collect', keys=['img']),
    ]

    anno = json.load(open(args.json))
    test_pipeline = Compose(test_pipeline)
    
    # build model
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
    model = TeViT()
    param_dict = mindspore.load_checkpoint(args.ckpt)

    mindspore.load_param_into_net(model, param_dict)
    model = mindspore.Model(network=model)

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
        
        metas = [_['img_metas'].data for _ in datas]

        def np2ms(x):
            if isinstance(x, list):
                for x_  in x:
                    np2ms(x_)
            elif isinstance(x, dict):
                for k in x.keys():
                    if isinstance(x[k], np.ndarray):
                        x[k] = mindspore.Tensor(x[k])
                    elif isinstance(x[k], dict):
                        np2ms(x[k])

        np2ms(metas)
        imgs = np.stack([_['img'] for _ in datas])
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        x = mindspore.Tensor(imgs.astype(np.float32))

        bboxes, labels, scores, masks = model.predict(x, metas)

        for inst_ind in range(bboxes.shape[1]):
            objs = dict(
                video_id=video['id'],
                score=scores[inst_ind],
                category_id=labels[inst_ind] + 1,
                segmentations=[])
            for sub_ind in range(masks.shape[0]):
                m = masks[sub_ind, inst_ind, ...].astype(np.uint8)
                m_ = mask_utils.encode(
                    np.array(m[:, :, np.newaxis], order='F', dtype='uint8'))[0]
                m_['counts'] = m_['counts'].decode()
                objs['segmentations'].append(m_)
            results.append(objs)

    json.dump(results, open('results.json', 'w'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
