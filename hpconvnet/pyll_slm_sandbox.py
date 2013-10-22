"""
Temporary for operations that are not originally included in the library.

"""
import logging

import numpy as np

import theano.tensor as tensor
from hyperopt import pyll
from .pyll_slm import boxconv


logger = logging.getLogger(__name__)

@pyll.scope.define_info(o_len=2)
def slm_lpool_rectlin((x, x_shp),
        ker_size=3,
        order=1,
        stride=1,
        ):
    """
    lpool with x = max(x,0) as first step
    """
    assert x.dtype == 'float32'
    order=float(order)

    ker_shape = (ker_size, ker_size)

    x = tensor.maximum(x, 0)
    r, r_shp = boxconv((x ** order, x_shp), ker_shape)
    r = r ** (1. / order)

    if stride > 1:
        # -- theano optimizations should turn this stride into conv2d
        #    subsampling
        r = r[:, :, ::stride, ::stride]
        # intdiv is tricky... so just use numpy
        r_shp = np.empty(r_shp)[:, :, ::stride, ::stride].shape

    assert r_shp[2] == r_shp[3]
    return r, r_shp
