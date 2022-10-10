# Revisiting Unsupervised Meta-Learning via the Characteristics of Few-Shot Tasks

The code repository for "Revisiting Unsupervised Meta-Learning via the Characteristics of Few-Shot Tasks" 

## Overview

We first analyze the factors to meta-train a UML method and propose SES and SNS as two key ingredients towards a strong UML baseline. Then, we propose HMS and TSP-Head to further utilize the characteristic of tasks from different aspects, which additionally improve either lower or higher shots scenarios.


<img alt="figure of method" src='imgs/method.png' width='640'>

## Unsupervised Meta Learning Results

Experimental results on few-shot learning datasets with ResNet-12 backbone (Same as [this repo](https://github.com/kjunelee/MetaOptNet)). We report average results with 10,000 randomly sampled few-shot learning episodes for stablized evaluation.

**MiniImageNet Dataset with ConvNet**

| (way,shot) |  (5,1) |  (5,5) | (5,20) | (5,50) |
|:----------:|:------:|:------:|:------:|:------:|
|  baseline  | 47.43  | 64.11  | 72.52  | 74.72  |
|  TSP-Head  | 47.35  | 65.10  | **74.45**  | **77.03**  |
|     HMS    | **48.12**  | **65.33**  | 73.31  | 75.49  |

**MiniImageNet Dataset with ResNet-12**

| (way,shot) |  (5,1) |  (5,5) | (5,20) | (5,50) | checkpoint                                                                                         |
|:----------:|:------:|:------:|:------:|:------:|----------------------------------------------------------------------------------------------------|
|  baseline  | 56.74  | 74.05  | 81.24  | 83.04  | [google drive](https://drive.google.com/file/d/1JLzVOUN-VSM2AesIorB8-tE0TsFol2aY/view?usp=sharing)                                                                                   |
|  TSP-Head  | 56.99  | **75.89**  | **83.77**  | **85.72**  | [google drive](https://drive.google.com/file/d/1ZTbmdPY5ClgLv8CAlVSonZ0HGjEUFHRO/view?usp=sharing) |
|     HMS    | **58.20**  | 75.77  | 82.69  | 84.41  | [google drive](https://drive.google.com/file/d/1ac9iPsUFAeZrkOGgBf_A2QOJyI12IYjS/view?usp=sharing) |

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images into the folder data/[name of the dataset, miniimagenet or cub]/images

## Dataset

### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation, respectively.

### CUB Dataset
[Caltech-UCSD Birds (CUB) 200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) is initially designed for fine-grained classification. It contains in total 11,788 images of birds over 200 species. On CUB, we randomly sampled 100 species as SEEN classes, and another two 50 species are used as two UNSEEN sets. We crop all images with given bounding boxes before training. We only test CUB with the ConvNet backbone in our work.

### TieredImageNet Dataset
[TieredImageNet](https://github.com/renmengye/few-shot-ssl-public) is a large-scale dataset  with more categories, which contains 351, 97, and 160 categoriesfor model training, validation, and evaluation, respectively. The dataset can also be download from [here](https://github.com/kjunelee/MetaOptNet).
We only test TieredImageNet with ResNet backbone in our work.

Check [this](https://github.com/Sha-Lab/FEAT/blob/master/data/README.md) for details of data downloading and preprocessing.

## Code Structures
To reproduce our experiments with FEAT, please use **train_fsl.py**. There are four parts in the code.
 - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.
 - `data`: Images and splits for the data sets.
 - `saves`: The pre-trained weights of different networks.
 - `checkpoints`: To save the trained models.
 - `script`: Example commands.

## Model Training and Evaluation
Please use **train.py** and follow the instructions below. FEAT meta-learns the embedding adaptation process such that all the training instance embeddings in a task is adapted, based on their contextual task information, using Transformer. The file will automatically evaluate the model on the meta-test set with 10,000 tasks after given epochs.

## Arguments
The train.py takes the following command line options (details are in the `model/utils.py`):

**Task Related Arguments**
- `dataset`: Option for the dataset (`MiniImageNet`, `TieredImageNet`,`CIFAR-FS` , `FC100` , or `CUB`), default to `MiniImageNet`

- `data_root`: root directory for the dataset.

- `way`: The number of classes in a few-shot task during meta-training, default to `5`

- `eval_way`: The number of classes in a few-shot task during meta-test, default to `5`

- `shot`: Number of instances in each class in a few-shot task during meta-training, default to `1`

- `eval_shot`: Number of instances in each class in a few-shot task during meta-test, default to `1`

- `query`: Number of instances in each class to evaluate the performance during meta-training, default to `15`

- `eval_query`: Number of instances in each class to evaluate the performance during meta-test, default to `15`

**Optimization Related Arguments**
- `max_epoch`: The maximum number of training epochs, default to `200`

- `episodes_per_epoch`: The number of tasks sampled in each epoch, default to `100`

- `num_eval_episodes`: The number of tasks sampled from the meta-val set to evaluate the performance of the model (note that we fix sampling 10,000 tasks from the meta-test set during final evaluation), default to `200`

- `lr`: Learning rate for the model, default to `0.0001` with pre-trained weights

- `lr_scheduler`: The scheduler to set the learning rate (`step`, `multistep`, or `cosine`), default to `step`

- `step_size`: The step scheduler to decrease the learning rate. Set it to a single value if choose the `step` scheduler and provide multiple values when choosing the `multistep` scheduler. Default to `20`

- `gamma`: Learning rate ratio for `step` or `multistep` scheduler, default to `0.2`

- `augment`: Whether to do data augmentation or not during meta-training, default to `False`

- `mom`: The momentum value for the SGD optimizer, default to `0.9`

- `weight_decay`: The weight_decay value for SGD optimizer, default to `0.0005`

**Model Related Arguments**

- `model_class`: The model to use. We provide implementations for uml baseline (`ProtoNet`) and `TSP-Head`. (Note that we implement HMS as an additional component, see `additional`.) Default to `ProtoNet`

- `similarity`: Similarity measure of two instances. We provide four implementation `euclidean` ,inner product (`dot`), `cosine` and `sns` . Default to `sns`

- `backbone_class`: Types of the encoder, i.e., the convolution network (`ConvNet`), ResNet-12 (`Res12`), or Wide ResNet (`WRN`), default to `ConvNet`

- `temperature`: Temperature over the logits, we #divide# logits with this value. It is useful when meta-learning with pre-trained weights. Default to `1`

- `temperature2`: Temperature over the logits in the regularizer, we divide logits with this value. This is specially designed for the contrastive regularizer. Default to `1`

- `additional`: Additional component that can add to model. Can be `HMS` or `none`. Default to `none`.

Arguments for TSP-Head:

- `lr_mul`: This is specially designed TSP-Head. The learning rate for the top layer will be multiplied by this value (usually with faster learning rate). Default to `10`

- `t_heads`: Number of heads for Multi-Head Transformer. Default to `8`.

- `t_layers`: Number of layers for Transformer. Default to `1`.

- `t_dropout`: Drop out rate for Transformer. Default to `0.2`.

- `t_dim`: Dimension of Keys and Values for Transformer. `-1` means equaling to dimension of input feature vector. Default to `-1`.

Arguments for HMS:

- `hard_negs` : Number of hard negative samples for each query instance. Default to `10`.

- `strength` : Mix up strength. The mix up factor will be draw from U(0,`strength`). Default to `0.5`.

**Other Arguments** 
- `orig_imsize`: Whether to resize the images before loading the data into the memory. `-1` means we do not resize the images and do not read all images into the memory. Default to `-1`

- `multi_gpu`: Whether to use multiple gpus during meta-training, default to `False`

- `gpu`: The index of GPU to use. Please provide multiple indexes if choose `multi_gpu`. Default to `0`

- `log_interval`: How often to log the meta-training information, default to every `50` tasks

- `eval_interval`: How often to validate the model over the meta-val set, default to every `1` epoch

- `save_dir`: The path to save the learned models, default to `./checkpoints`

Running the command without arguments will train the models with the default hyper-parameter values. Loss changes will be recorded as a tensorboard file.

## Training scripts

For example, to train UML baseline with ResNet-12 backbone on MiniImageNet:

    $ python train.py --eval_all --unsupervised --batch_size 32 --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 --dataset MiniImageNet --way 5 --shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.03 --lr_scheduler cosine --gpu 0 --eval_interval 2 --similarity sns
    
to train TSP-Head with ResNet-12 backbone on MiniImageNet
    
    $ python train.py --eval_all --unsupervised --batch_size 32 --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class TSPHead --backbone_class Res12 --dataset MiniImageNet --way 5 --shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.03 --lr_mul 10 --lr_scheduler cosine --gpu 0 --eval_interval 2 --similarity sns --t_heads 8
    
to train HMS with ResNet-12 backbone on MiniImageNet

    $ python train.py --eval_all --unsupervised --batch_size 32 --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 --dataset MiniImageNet --way 5 --shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.03 --lr_mul 1 --lr_scheduler cosine --gpu 0 --eval_interval 2 --similarity sns --additional HMS --strength 0.5
 
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)

- [MatchingNet](https://github.com/gitabcworld/MatchingNetworks)

- [PFA](https://github.com/joe-siyuan-qiao/FewShot-CVPR/)

- [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

- [MetaOptNet](https://github.com/kjunelee/MetaOptNet/)

- [FEAT](https://github.com/Sha-Lab/FEAT)