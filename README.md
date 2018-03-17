# Extra-Chainer
Useful classes and functions implementation for Chainer, the deep learning framework.

# Requirements
- [Chainer (test only on ver.4.0.0b3)](https://github.com/pfnet/chainer) (Neural network framework)

# links
Implementations of chainer.Link
- separable_link
  - Wrapper classes for making chainer 'link' separable.
  - For example, create channel separable convolution from links.Convolution2D.

# functions
Implementations of chainer.Function
- exadd
  - Extra addition function for Variables they have mismatch shapes.
  - For example, it is able to used for merging branches with different channels in ResNetA.

# Usages
Please look in each directory.
