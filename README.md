# chainer-my-links
Layer classes implementation for deep learning by Chainer.

# Requirements
- [Chainer (test only on ver.4.0.0b3)](https://github.com/pfnet/chainer) (Neural network framework)

# separable_link
Wrapper classes for making chainer 'link' separable,
for example, create channel separable convolution from links.Convolution2D.
In python script, write:
```
import chainer
import chainer.links as L
from separable_link import SeparableLink, SeparableSampleLink

class MyCnnModel(chainer.Chain):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        with self.init_scope()
            ...
            # separable convolution.
            self.sepA = SeparableLink(link=L.Convolution2D, axis=1, n=8)
            # separable convolution on the axis of sample. It need reduced on test phase.
            self.sepB = SeparableSampleLink(link=L.Convolution2D, _reduce=sum, _normalize=True, n=4)
            ...

    def __call__(self, x):
        ...
        x = self.sepA(x)
        x = self.sepB(x)
        ...
        return x
```
