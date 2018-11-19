# functions
Implementations of chainer.Function

# Usage
## top_k_accuracy
top-k accuracy implementation for [ImageNet](http://www.image-net.org/challenges/LSVRC/) as chainer function.  
Use `Classifier(..., accfun=top_k_accuracy)` or `Clasifier(..., accfun=get_top_k_accuracy_func(top_k=3)`.  

## mixfeat
Re-Implementation of this paper! -> [MixFeat: Mix Feature in Latent Space Learns Discriminative Space](https://openreview.net/forum?id=HygT9oRqFX)

## shuffle
Shuffle array implementation for [shuffleNet](https://arxiv.org/abs/1707.01083) and/or [shuffleNet V2](https://arxiv.org/abs/1807.11164) as chainer function.  

## perturb
Re-Implementation of this paper. -> [Perturbative Neural Networks](https://arxiv.org/abs/1806.01817)

## pgp
Re-Implementation of this paper! -> [Parallel Grid Pooling for Data Augmentation](https://github.com/akitotakeki/pgp-chainer)

## exadd
Extra addtion function for Variables they have mismatch shapes.
For example, it is able to used for merging branches with different channels in ResNetA or PyramidNet.  
In python script, write:
```
import chainer
from exadd import exadd

class MyCnnModel(chainer.Chain):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        with self.init_scope()
            ...

    def __call__(self, x):
        ...
        p = exadd((q, r)) # output shape is same as q.shape
        ...
        w = exadd((x, y, z), shape=given_shape) # output shape is same as given_shape
        ...
        a = exadd_maxshape((b, c, d)) # output shape is circumscribed shape of (b, c, d)
        ...
        return x
```
## shake_drop
[shake-drop](https://arxiv.org/abs/1802.02375) implementation as chainer function.  
In python script, write:
```
import chainer
from shake_drop import ShakeDrop

class MyCnnModel(chainer.Chain):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        with self.init_scope()
            ...

    def __call__(self, x):
        ...
        p = ShakeDrop(x) # shake-drop, on the pixel-level, and depthwise stochastistic (probability 0.5 on the last layer).
        ...
        return x
```
