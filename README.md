# Spatial-Temporal Differential Evolution Patch Attack
Paper code: “[Efficient Decision-based Black-box Patch Attacks on Video Recognition](https://arxiv.org/abs/2303.11917)”.

## Start
```
git clone https://github.com/deep-kaixun/STDE.git
cd STDE
conda env create -f environment.yaml
```

# Model
C3D are from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch). NL and TPN models are from [GluonCV](https://cv.gluon.ai/model_zoo/action_recognition.html)

## Pretrained Model
Download [here](https://drive.google.com/open?id=1DmI6QBrh7xhme0jOL-3nEutJzesHZTqp).

## Untargeted Attacks
```bash
python3 attack.py --batch_size 1 --cf_rate 0.6 --time_mua 1
```

## Targeted Attacks
```bash
python3 attack.py --batch_size 1 --cf_rate 0.7 --time_mua 2 --targeted
```

## Acknowledgements
BSCA [Paper](https://arxiv.org/abs/2110.15629) | [Code](https://github.com/kay-ck/BSC-Attack)




# Citation
If you use the code or find this project helpful, please consider citing our paper.
```
@article{jiang2023efficient,
  title={Efficient Decision-based Black-box Patch Attacks on Video Recognition},
  author={Jiang, Kaixun and Chen, Zhaoyu and Huang, Hao and Wang, Jiafeng and Yang, Dingkang and Li, Bo and Wang, Yan and Zhang, Wenqiang},
  journal={arXiv preprint arXiv:2303.11917},
  year={2023}
}
```
