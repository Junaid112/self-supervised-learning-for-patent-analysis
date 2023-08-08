# self-supervised-learning-for-patent-analysis

## Abstract
The core of this approach is by using three different self-supervised learning method, i.e. Bootstrap Your Own Latent (BYOL), Simple Contrastive Learning of Representations(SimClR) and Momentum Contrast(Moco) models to accomplish three tasks: image type classification, perspective image classification and object image classification. Through the experiments, we can prove that self-supervised learning can provide a choice for patent analysis. 

## Self-supervised training part

In this part we use Resnet50 as our encoder and train the model with three different self-supervised learning methods. The code of SimCLR refers to:
https://blog.csdn.net/qq_43027065/article/details/118657728. The code of Moco can be found in https://github.com/facebookresearch/moco. After the training process, the encoder will be saved to use for downstream tasks, i.e. validation part.

## Validation part

In this part the encoder is connected to a linear classifier. During the training process, all the layers in encoder is freezen. Because the training and test datasets are not so big, to avoid overfitting problems, the linear classifier isn't so deep. After the activation function Softmax, we can get a 1*n vector which is the possibilities for each class.

## Hierarchical image classification

For perspective image classification we build a tree structure with three stages. So for this task we also try to use hierarchical image classification. We select the C-HCMNN model and the code refers to https://github.com/EGiunchiglia/C-HMCNN.

