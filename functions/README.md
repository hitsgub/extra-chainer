# links
Implementations of chainer.Function

# Usages
## exadd
Extra addtion function for Variables they have mismatch shapes.
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
        return x
```
