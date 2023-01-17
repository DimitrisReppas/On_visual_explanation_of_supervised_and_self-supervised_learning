# On visual explanation of supervised and self-supervised learning

This project is part of my MSc Thesis https://pergamos.lib.uoa.gr/uoa/dl/frontend/el/browse/3256221  for my postgraduate studies in Data Science and Information Technologies (NKUA).

The primary objective of this project is to interpret both supervised and self-supervised models, using either *convolutional neural networks* or *visual transformers* as a backbone. Variations of visualization methods are used, based on *class activation maps* (CAM) and *attention* mechanisms. Given an input image, these methods provide us with a *saliency map* that is used to interpret the network prediction. This map indicates the regions of the image that the model pays the most attention to. We evaluate these methods qualitatively and quantitatively. We further propose new alternative or complementary visualization methods, which show where important information can be hidden inside the network and how to reveal it. These new methods further improve the quantitative results. Our study highlights the importance of *interpretability*, shows some common properties and differences in the way supervised and self-supervised models make their predictions and provides valuable information on both the models and the visualization methods.


## Backbone architectures

- [ResNet 50](https://arxiv.org/abs/1512.03385)
- [DeiT](https://arxiv.org/abs/2012.12877)

## Self-supervised approaches

- [DINO](https://arxiv.org/abs/2104.14294) 

- [MoCo v3](https://arxiv.org/abs/2003.04297)
  
The weights for ResNet-based and DeiT-based models trained with MoCo v3 approach on ImageNet can be found https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md.

Download them and put them into: /path/to/On_visual_explanation_of_supervised_and_self-supervised_learning/storage/pretrained_weights/

## Visualization Methods
Methods based on CAM
- [GradCAM](https://arxiv.org/abs/1610.02391)
- [GradCAM++](https://arxiv.org/abs/1710.11063)
- [XGradCAM](https://arxiv.org/abs/2008.02312)
- [Score-CAM](https://arxiv.org/abs/1910.01279) 

Attention-based Methods
- Raw attention maps from different layers
- [Rollout](https://arxiv.org/abs/2005.00928)

## Dataset

The dataset we use for the experiments is a subset of the validation set of ImageNet and can be found https://github.com/EliSchwartz/imagenet-sample-images. 

Download it and put it into: /path/to/On_visual_explanation_of_supervised_and_self-supervised_learning/storage/imagenet_1000samples/

## Run Locally

Clone the project

```bash
  git clone https://github.com/DimitrisReppas/On_visual_explanation_of_supervised_and_self-supervised_learning.git
```

## Install dependencies
Install the requirements for this project

```bash
  pip install -r /path/to/On_visual_explanation_of_supervised_and_self-supervised_learning/storage/requirements.txt
```

## Evaluate models based on ResNet

Example
- Evaluate ResNet 50 qualitatively and quantitatively using GradCAM with:

```bash
  python Res_based_evaluation.py --use-cuda --method gradcam --arch res_50 --data_path ../storage/imagenet_1000samples/ --output_dir ../storage/quant_results/ --saliency True
```

## Evaluate models based on DeiT

Example
- Evaluate MoCo v3 that uses DeiT as backbone quantitatively using Rollout method with:

```bash
  python Deit_based_evaluation.py --use-cuda --method rollout --arch moco_v3_deit_base --data_path ../storage/imagenet_1000samples/ --output_dir ../storage/quant_results/ --saliency False
```

## Results
Bellow, some qualitative and quantitative results are presented:
- Saliency maps for a given input image, obtained from different models that are based on ResNet 50, when using four different CAM-based methods

![qual_res](https://user-images.githubusercontent.com/74918841/212973589-bad8258d-a88c-4370-a4b9-9e9a63dcf658.png)

- Qualitative and quantitative results for all ResNet-based models, when using the GradCAM method

![qual_quant_res](https://user-images.githubusercontent.com/74918841/212977958-afe13f21-49bc-46ac-973a-a8afc5d5b672.png)

- Quantitative evaluation of CAM-based saliency maps for models using ResNet as a backbone

![quant_res](https://user-images.githubusercontent.com/74918841/212978765-a7bd1363-fa84-4264-b3c2-dd739effd757.png)

- Visualization of Raw attention maps derived from different layers and from models that are using DeiT as a backbone

![qual_raw_deit](https://user-images.githubusercontent.com/74918841/212979452-98815efa-e5f0-4a56-9418-65e34b96c3f4.png)

- Qualitative results for DeiT-based models when the Rollout method and Raw attention maps, derived from the last layer, are used

![qual_raw_rollout](https://user-images.githubusercontent.com/74918841/212981569-09ac8f00-ff7a-456e-87b2-02edbbfca406.png)

- Quantitative results for DeiT-based models when Raw attention maps, derived from the last layer, are used.

![quant_raw](https://user-images.githubusercontent.com/74918841/212982076-7bcdeaec-b2b4-4a1e-85f9-c9913a8a6e3b.png)
