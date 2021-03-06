# Generative Dynamic Patch Attack

This reposityory contains the PyTorch implementation of our paper "Generative Dynamic Patch Attack".

[comment]: # (<img src="https://github.com/lxuniverse/gdpa/blob/main/pics/gdpa_arch.png" width="600" class="center">)

## Requirements
PyTorch >= 1.6.0

TensorBoard >= 2.2.1

tqdm

## Quick Start
Download the data and CE trained model of VGGFace from:

https://github.com/tongwu2020/phattacks/releases/tag/Data%26Model

Download the data of ImageNet from:

http://www.image-net.org/ 

---

1. Dynamic patch attack with GDPA:
```
python gdpa.py --dataset [imagenet|vggface] --data_path [FOLDER_NAME] 

If on VGGFace, please add --vgg_model_path [MODEL_PATH]

optional arguments:
  --patch_size            size of adversarial patch
  --alpha                 $\alpha$ in paper
  --beta                  $\beta$ in paper
  --exp                   exp name in logging
  --epochs                epochs for training
  --lr_gen                learning rate
  --batch_size            batch size
  --device                cuda or cpu
```
2. Adversarial training with GDPA-AT:
```
python gdpa_at.py --data_path [FOLDER_NAME] --vgg_model_path [MODEL_PATH] 

optional arguments:
  --patch_size            size of adversarial patch
  --beta                  $\beta$ in paper
  --lr_gen                learning rate for generator
  --lr_clf                learning rate for classifier
  --save_freq             frequency of saving the model
  --epochs                epochs for training
  --batch_size            batch size
  --device                cuda or cpu
  --enable_testing        testing during training
```
3. Visulize ASRs and adversarial images with tensorboard:
```
tensorboard --logdir logs/exp/gdpa/
```
```
tensorboard --logdir logs/exp/gdpa_at/
```

## Citation

If you find this repository useful, please cite our paper:
```
@article{xiang2021gdpa,
    title={Generative Dynamic Patch Attack},
    author={Xiang Li and Shihao Ji},
    journal={British Machine Vision Conference (BMVC)},
    year={2021}
}
```
