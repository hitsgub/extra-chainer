# Extra-Chainer
Useful classes and functions implementation for Chainer, the deep learning framework.

# Requirements
- [Chainer (test only on ver.4.0.0b3)](https://github.com/pfnet/chainer) (Neural network framework)

# links
Implementations of chainer.Link
- chain_modules
  - CNN module definer by array of keys such as 'I+CBRCB>R',  
    where I=identity-mapping, B=BN, R=ReLU, C=Conv, etc...,  
    '+'=additional join, ','=concatenation join, '>'=sequential join.
  - And, you can add keys for your own new methods.
- network_modules
  - CNN Encoder definer using 'chain_modules.Module'.
In python script, write:
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
Implementations of neural network models by chainer.Link

# utils
Implementations of utility functions.
- model_info
  - Getter of string of model informations.  
    For example, number of links, number of weights, number of parameters.

# Usages
Please look in each directory.
