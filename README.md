# Generative Dynamic Patch Attack

This reposityory contains the PyTorch implementation of "Generative Dynamic Patch Attack".

[comment]: # (<img src="https://github.com/lxuniverse/gdpa/blob/main/pics/gdpa_arch.png" width="600" class="center">)

## Requirements
PyTorch >= 0.4.0

## Quick Start
Download the data from:

https://github.com/tongwu2020/phattacks for VGGFace

or

http://www.image-net.org/ for ImageNet

---

1. Train GDPA for patch attack:
```
python gdpa.py --size 32 --alpha 1 --beta 3000 --dataset imagenet --data_path [folder of imagenet]
```
2. Train a robust model with GDPA-AT
```
python gdpa_at.py --data_path [folder of vggface]
```
