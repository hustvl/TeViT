# TeViT on MindSpore

## Installation

1. create and activate python enviroment
```python
conda create --name tevit python=3.8
conda activate tevit
```

2. install mindspore
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.10.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.10.1-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Download Model Weights

We provide the basic TeViT-Res50 in [here]().

## Inference

```bash
python test_vis.py --json /PATH/TO/JSON --root /PATH/TO/ROOT --ckpt /PATH/TO/CKPT
```
After inference process, the predicted results is stored in ```results.json```, submit it to the evaluation server to get the final performance.
