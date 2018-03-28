# functions
Implementations of chainer.Function

# Usages
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
[shake-drop](https://arxiv.org/abs/1802.02375) implementation.  
We call the shake-drop implementation as one of the layers.
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
        p = ShakeDrop(x) # shake-drop, on the axes of (0, 1, 2, 3), and depthwise stochastistic (pL=0.5).
        ...
        return x
```
