## DDSPSeg

<p align="center"><img width="100%" src="fig/name1.jpg" /></p>

[NEWS!]**This paper has been accpeted by [MIA 2024](https://www.sciencedirect.com/journal/medical-image-analysis)! The basic code on [PyTorch](https://github.com/MIXAILAB/DDSPSeg) has been opened!** 

[NOTE!!]**The code will be gradually and continuously opened!**

Recent unsupervised domain adaptation (UDA) methods in medical image segmentation commonly utilize Generative Adversarial Networks (GANs) for domain translation. However, the translated images often exhibit a distribution deviation from the ideal due to the inherent instability of GANs, leading to challenges such as visual inconsistency and incorrect style, consequently causing the segmentation model to fall into the fixed wrong pattern. To address this problem, we propose a novel UDA framework known as Dual Domain Distribution Disruption with Semantics Preservation (DDSP). Departing from the idea of generating images conforming to the target domain distribution in GAN-based UDA methods, we make the model domain-agnostic and focus on anatomical structural information by leveraging semantic information as constraints to guide the model to adapt to images with disrupted distributions in both source and target domains. Furthermore, we introduce the inter-channel similarity feature alignment based on the domain-invariant structural prior information, which facilitates the shared pixel-wise classifier to achieve robust performance on target domain features by aligning the source and target domain features across channels.

<p align="center"><img width="100%" src="fig/name2.jpg" /></p>

## Paper
This repository provides the official PyTorch implementation of DDSPSeg in the following papers:

**[Dual domain distribution disruption with semantics preservation: Unsupervised domain adaptation for medical image segmentation](https://doi.org/10.1016/j.media.2024.103275)** <br/> 
[Boyun Zheng, Ranran Zhang, Songhui Diao, Jingke Zhu, Yixuan Yuan, Jing Cai, Liang Shao, Shuo Li, Wenjian Qin] <br/>
University of Chinese Academy of Sciences <br/>

## Dataset
 * Multi-Modality Whole Heart Segmentation Challenge 2017 (MMWHS17)
 * Multi-Modality Brain Tumor Segmentation Challenge 2018 (BraTS18)
 * PROMISE12 challenge (Pro12) dataset

 More information and downloading links of the former three datasets can be found in [MMWHS17](https://zmiclab.github.io/zxh/0/mmwhs/), [BraTS18](https://www.med.upenn.edu/sbia/brats2018/data.html) and [Pro12](https://liuquande.github.io/SAML/).

## Training from Scratch 
Datasets are like this:
```bash
dataset/
│
├── domainA/
    ├── patient1
        ├── image.nii
        ├── label.nii
    ├── patient2
├── domainB/
    ├── patient1
        ├── image.nii
        ├── label.nii
    ├── patient2
```
```bash
python train.py
```

## Testing 
```bash
python test.py
```

## Citation
If you use this code for your research, please cite our papers:
```
@article{ZHENG2024103275,
title = {Dual domain distribution disruption with semantics preservation: Unsupervised domain adaptation for medical image segmentation},
journal = {Medical Image Analysis},
volume = {97},
pages = {103275},
year = {2024},
issn = {1361-8415},
author = {Boyun Zheng and Ranran Zhang and Songhui Diao and Jingke Zhu and Yixuan Yuan and Jing Cai and Liang Shao and Shuo Li and Wenjian Qin}
}
```

## Acknowledgments

This research was partially funded by the National Natural Science Foundation of China (No. U20A20373), the Youth Innovation Promotion Association CAS, China (2022365) and Interventional Therapy Clinical Medical Research Center of Jiangxi Province, China (No.20223BCG74005).
