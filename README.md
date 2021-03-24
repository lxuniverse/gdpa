# Generative Dynamic Patch Attack

This reposityory contains the PyTorch implementation of "Generative Dynamic Patch Attack".

[comment]: # (<img src="https://github.com/lxuniverse/gdpa/blob/main/pics/gdpa_arch.png" width="600" class="center">)

## Requirements
PyTorch 
TensorBoard

## Quick Start
Download the data and CE trained model of VGGFace from:

https://github.com/tongwu2020/phattacks

Download the data of ImageNet from:

http://www.image-net.org/ 

---

1. Train GDPA for patch attack:
```
python gdpa.py --dataset [imagenet|vggface] --data_path [FOLDER_NAME] --vgg_model_path [MODEL_PATH]
optional arguments:
  --patch_size          
  --alpha
  --beta
```
2. Train a robust model with GDPA-AT:
```
python gdpa_at.py --data_path [FOLDER_NAME] --vgg_model_path [MODEL_PATH] 
optional arguments:
  --enable_testing     
```
3. Visulize logging ASRs and Images:
```
tensorboard --logdir logs/exp/gdpa/
```
```
tensorboard --logdir logs/exp/gdpa_at/
```
