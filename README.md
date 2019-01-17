# Extra-Chainer
Useful classes and functions implementation for Chainer, the deep learning framework.  
Various novel methods are (and will be) implemented for examples,  
and various CNN-models are available in .\models\ directory.
- .\models\
  - [ShakeDrop](https://github.com/imenurok/ShakeDrop)
  - [PGP: Parallel Grid Pooling](https://github.com/akitotakeki/pgp-chainer)  
    * Use with .\links\multiplex_classifier.py.
  - [Perturbative Neural Networks](https://arxiv.org/abs/1806.01817)
  - PGPflip, extention of PGP.
  - FlipAugmentation, inspired from PGP.
  - [SE-Net](https://arxiv.org/abs/1709.01507)  
    * PreResNet20_SE.py
  - [competitive SE-Net](https://arxiv.org/abs/1807.08920)  
    * PreResNet20_CSE_WFC.py (double-FC version in original paper.)  
    * PreResNet20_CSE.py (modified implementation)  
  - [ShuffleNet V2](https://arxiv.org/abs/1807.11164)  
    * PreResNet20_sv2.py (modified implementation for CIFAR)  
    * PreResNet20_dwc_sv2.py (with depthwiseconv, modified implementation for CIFAR)  
  - [MixFeat](https://openreview.net/forum?id=HygT9oRqFX)  
    * PreResNet20_mixfeat.py
  - [FishNet](https://arxiv.org/abs/1901.03495)  
    * Fish* Net*.py (modified implementation for CIFAR)  
    * Fish* Mix*.py (modified implementation for CIFAR, with MixFeat)  
- .\iterators\  
  - [Between Class Learning](https://github.com/mil-tokyo/bc_learning_image)  
    - BCIterator.py  
      * Use with .\functions\accuracy_mix.py and .\functions\kl_divergence.py.  
    The code is refactor from original code for reusability.

## Requirements
- [Chainer (test on ver.6.0.0)](https://github.com/pfnet/chainer) (Neural network framework)
- [ChainerCV (test on ver.0.8.0)](https://github.com/chainer/chainercv) (a Library for Deep Learning in Computer Vision)
- [chainer-computational-cost](https://github.com/belltailjp/chainer_computational_cost) (a tool to estimate theoretical computational cost in chainer)

## Training example
```
# ordinary learning
python trainer.py --gpu [# of GPU] --model [ex.)models\PreResNet20.py]
# between class learning
python trainer_bcl.py --gpu [# of GPU] --model [ex.)models\PreResNet20.py]
```
- More information is in .\get_arguments.py

## Send message to slack example
1. Get slack token and save as `slack_token` on root directory.  
  To get token, please show [here](https://qiita.com/yuishihara/items/2782a76affb5fa574349), thanks to @yuishihara.  
2. Run below command. The option `slack_interval` means interval of epoch.  
   *The example send message to slack channel `bot`.  
    If you want to send to other channel, you rewrite `SlackOut()` to `SlackOut(channel='xxx')` in `trainer.py`.  
```
python trainer.py --gpu [# of GPU] --model [ex.)models\PreResNet20.py] --slack_interval 10
```

## links
Implementations of chainer.Link
- chain_modules
  - CNN module definer by array of keys such as 'I+CBRCB>R',  
    where I=identity-mapping, B=BN, R=ReLU, C=Conv3x3, c=Conv1x1, etc...,  
    '*'=productional connect, '+'=additional connect, ','=concatenation(axis=1) connect,  
    '|'=concatenation(axis=0) connect, '>'=sequential connect,  
    and 'integer' for example 2 or 4, denotes channel scaling factor.
  - And, you can add keys for your own new methods.
- network_modules
  - CNN Encoder definer using 'chain_modules.Module'.

In python script, write chain_modules and network_modules:
```
from chain_modules import Module
from network_modules import Encoder
import chainer.links as L

class MyCnnModel(chainer.Chain):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        with self.init_scope()
            ...
            # ResNet module definition
            self.res = Module(16, 32, 'I+CBRCB>R')
            # PreActResNet module definition
            self.pres = Module(16, 32, 'I+BRCBRC')
            # PreActResNet (bottleneck) module definition
            self.bres = Module(16, 32, 'I+BRcBRcBR4c')
            # ResNeXt module definition
            self.resx = Module(64, 128, 'I+BR8cBR8GBR4c', G=(lambda s: L.Convolution2D(None, s.ch, 3, s.stride, 1, group=8)))
            # DenseNet module definition
            self.dense = Module(16, 12, 'I,BRC')
            # Encoder part of ResNet20 definition
            self.res20 = Encoder((3, 3, 3), None, (16, 32, 64), 'I+CBRCB>R', 'A', None, (1, 1, 1))
            ...
```

- separable_link
  - Wrapper classes for making chainer 'link' separable.  
    For example, create channel separable convolution from links.Convolution2D.

## functions
Implementations of chainer.Function
- exadd
  - Extra addition function for Variables they have mismatch shapes.  
    For example, it is able to used for merging branches with different channels in ResNetA.

## models
Implementations of neural network models by chainer.Link.
- network_templates
  - template of neural network models,  
    for example, ResNet, PyramidNet, DenseNet, DenseNet.

Various examples are available in the directory.

## utils
Implementations of utility functions.
- model_info
  - Getter of string of model informations.  
    For example, number of links, number of weights, number of parameters.
- utils.attention_shape(axes, shape)
  - get modified shape, remain length of axis in axes, reduce to '1' length of axis not in axes.  
    For example, utils.attention((1, 2), (10, 20, 30, 40)) -> (1, 20, 30, 1).

## Usages
Please look in each directory.
