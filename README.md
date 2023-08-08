# self-supervised-learning-for-patent-analysis

## Abstract
The core of this approach is by using three different self-supervised learning method, i.e. Bootstrap Your Own Latent (BYOL), Simple Contrastive Learning of Representations(SimClR) and Momentum Contrast(Moco) models to accomplish three tasks: image type classification, perspective image classification and object image classification. Through the experiments, we can prove that self-supervised learning can provide a choice for patent analysis. 

## Self-supervised training

In this part we use Resnet50 as our encoder and train the model with three different self-supervised learning methods. The code of SimCLR refers to:
https://blog.csdn.net/qq_43027065/article/details/118657728

