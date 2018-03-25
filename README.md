# Extra-Chainer
Useful classes and functions implementation for Chainer, the deep learning framework.

# Requirements
- [Chainer (test only on ver.4.0.0b3)](https://github.com/pfnet/chainer) (Neural network framework)

# links
Implementations of chainer.Link
- chain_modules
  - CNN module definer by array of keys such as 'I+CBRCB>R',  
    where I=identity-mapping, B=BN, R=ReLU, C=Conv3x3, c=Conv1x1, etc...,  
    '+'=additional join, ','=concatenation join, '>'=sequential join,
    and 'integer' for example 2 or 4, denotes channel scaling factor.
　- And, you can add keys for your own new methods.
- network_modules
  - CNN Encoder definer using 'chain_modules.Module'.

In python script, write chain_modules and network_modules:
```
from chain_modules import Module
from network_modules import Encoder
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
            self.bres = Module(16, 32, 'I+BRcBRcBR4c'
            # DenseNet module definition
            self.dense = Module(16, 12, 'I,BRC')
            # Encoder part of ResNet20 definition
            self.res20 = Encoder((3, 3, 3), None, (16, 32, 64), 'I+CBRCB>R', 'A', None, (1, 1, 1))
            ...
```

- separable_link
  - Wrapper classes for making chainer 'link' separable.  
    For example, create channel separable convolution from links.Convolution2D.

# functions
Implementations of chainer.Function
- exadd
  - Extra addition function for Variables they have mismatch shapes.  
    For example, it is able to used for merging branches with different channels in ResNetA.

# models
Implementations of neural network models by chainer.Link.
- network_templates
  - template of neural network models,  
    for example, ResNet, PyramidNet, DenseNet, DenseNet.

Various examples are available in the directory.

# utils
Implementations of utility functions.
- model_info
  - Getter of string of model informations.  
    For example, number of links, number of weights, number of parameters.
- utils.attention_shape(axes, shape)
  - get modified shape, remain length of axis in axes, reduce to '1' length of axis not in axes.  
    For example, utils.attention((1, 2), (10, 20, 30, 40)) -> (1, 20, 30, 1).

# Usages
Please look in each directory.
