<h2 align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/10794598">
    HEOI: Human Attention Prediction in Natural Daily Life with Fine-Grained Human-Environment-Object Interaction Model
  </a>
</h2>
<h4 align="center" color="A0A0A0"> Zhixiong Nan, Leiyu Jia, Bin Xiao* </h4>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

# HEOI
This is the official implementation of the paper "[HEOI: Human Attention Preiction in Natural Daily Life with Fine-Grained Human-Environment-Object Interaction Model](https://ieeexplore.ieee.org/abstract/document/10794598)".

<div align="center">
  <img src="figures/method.png"/>
</div><br/>

This paper handles the problem of human attention prediction in natural daily life from the third-person view. Due to the significance of this topic in various applications, researchers in the computer vision community have proposed many excellent models in the past few decades, and many models have begun to focus on natural daily life scenarios in recent years. However, existing mainstream models usually ignore a basic fact that human attention is a typical interdisciplinary concept. Specifically, the mainstream definition is direction-level or pixel-level, while many interdisciplinary studies argue the object-level definition. Additionally, the mainstream model structure converges to the dual-pathway architecture or its variants, while the majority of interdisciplinary studies claim attention is involved in the human-environment interaction procedure. Grounded on solid theories and studies in interdisciplinary fields including computer vision, cognition, neuroscience, psychology, and philosophy, this paper proposes a fine-grained Human-Environment-Object Interaction **HEOI** model, which for the first time integrates multi-granularity human cues to predict human attention. Our model is explainable and lightweight, and validated to be effective by a wide range of comparison, ablation, and visualization experiments on two public datasets.

## Update
[2025/3] Code for [HEOI](https://github.com/CQU-ADHRI-Lab/HEOI) is available here!

[2024/11] HEOI has been accepted at TIP as a regular paper!

## Installation

We tested our code with `Python=3.9.19, PyTorch=1.13.0, CUDA=11.4`. Please install PyTorch first according to [official instructions](https://pytorch.org/get-started/previous-versions/).

Example conda environment setup：

```bash
# Create a new virtual environment
conda create -n heoi python=3.9 -y
conda activate heoi

# Install PyTorch
pip install torch==1.13.0+cu114 torchvision==0.14.0+cu114 --extra-index-url https://download.pytorch.org/whl/cu113

# Under your working directory
git clone https://github.com/CQU-ADHRI-Lab/HEOI.git
cd MI-DETR/
pip install -r requirements.txt

# build an editable version of detrex
pip install -e .
```

## Models

<table style="width: 100%; border-collapse: collapse;"  id="model-table">
    <tr style="border: 1px solid black; background-color: #f2f2f2; text-align: center; padding: 8px;">
        <th align="center">Name</th>
        <th align="center">Backbone</th>
        <th align="center">Epochs</th>
        <th align="center"><i>AP</i></th>
        <th align="center">Download</th>
    </tr>
    <tr align="center">
        <td align="center"><a href="./projects/midetr/configs/midetr-resnet/midetr_r50_4scale_12ep.py" style="text-decoration: none; color: black;">MI-DETR</a></td>
        <td align="center">ResNet50</td>
        <td align="center">12</td>
        <td align="center">50.2</td>
        <td align="center"><a href="https://drive.google.com/file/d/1ONmGGOWcj4uzFjfrAQ9_H9ge8UhpdVxw/view?usp=drive_link" style="text-decoration: none; color: blue;">model</a></td>
    </tr>
    <tr align="center">
        <td align="center"><a href="./projects/midetr/configs/midetr-resnet/midetr_r50_4scale_24ep.py" style="text-decoration: none; color: black;">MI-DETR</a></td>
        <td align="center">ResNet50</td>
        <td align="center">24</td>
        <td align="center">51.2</td>
        <td align="center"><a href="https://drive.google.com/file/d/1FO1ht5N44clB1_65w5WQoUqB1COoM-lJ/view?usp=drive_link" style="text-decoration: none; color: blue;">model</a></td>
    </tr>
    <tr align="center">
        <td align="center"><a href="./projects/midetr/configs/midetr-swin/midetr_swin_large_384_4scale_12ep.py" style="text-decoration: none; color: black;">MI-DETR</a></td>
        <td align="center">Swin-Large-384</td>
        <td align="center">12</td>
        <td align="center">57.5</td>
        <td align="center"><a href="https://drive.google.com/file/d/1pCEOIIJ_jrQPWDxdWV8pxV-ODATFlKol/view?usp=drive_link" style="text-decoration: none; color: blue;">model</a></td>
    </tr>
</table>

## Run

### Training

Train MI-DETR with 8 GPUs:

```sh
python tools/train_net.py --config-file projects/midetr/configs/midetr-resnet/midetr_r50_4scale_12ep.py --num-gpus 8 --resume
```

### Evaluation

You can download our pretrained models and evaluate them with the following commands. 
```sh
python tools/train_net.py --config-file /path/to/config_file --num-gpus 8 --eval-only train.init_checkpoint=/path/to/model_checkpoint
```
For example, to reproduce our result, you can copy the config path from the model table, download the pretrained checkpoint into `/path/to/checkpoint_file`, and run 
```sh
python tools/train_net.py --config-file projects/midetr/configs/midetr-resnet/midetr_r50_4scale_12ep.py --num-gpus 8 --eval-only train.init_checkpoint=/path/to/model_checkpoint
```


## <a name="CitingMIDETR"></a>Citing HEOI

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@article{nan2024heoi,
  title={HEOI: Human Attention Prediction in Natural Daily Life With Fine-Grained Human-Environment-Object Interaction Model},
  author={Nan, Zhixiong and Jia, Leiyu and Xiao, Bin},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
```
