from .ensemble import Ensemble
from .ensemble_array import EnsembleArray
from .ensemble_scalar import EnsembleScalar

import numpy as np

from collections import OrderedDict

def ensemble_dot(a, b):
    """ Performs numpy.dot on every block if both inputs share same ensemble
    
    Arguments:
       a (EnsembleScalar or EnsembleArray): first operand
       b (EnsembleScalar or EnsembleArray): second operand
    """
    if a.ensemble != b.ensemble:
        raise ValueError("a and b are not defined on the same Ensemble")
    ensemble = a.ensemble

    if not isinstance(a, EnsembleScalar) and not isinstance(a, EnsembleArray):
        raise ValueError("a is neither of type EnsembleScalar nor EnsembleArray")

    if not isinstance(b, EnsembleScalar) and not isinstance(b, EnsembleArray):
        raise ValueError("b is neither of type EnsembleScalar nor EnsembleArray")

    data = OrderedDict()

    # both a and b are scalars
    if isinstance(a, EnsembleScalar) and isinstance(b, EnsembleScalar):
        for block, aa, bb in zip(a.keys(), a.values(), b.values()):
            data[block] = aa * bb
        return EnsembleScalar(ensemble, data)
    else:
        for block, aa, bb in zip(a.keys(), a.values(), b.values()):
            data[block] = np.dot(aa, bb)
        return EnsembleArray(ensemble, data)

def ensemble_asscalar(a):
    """ Transforms an EnsembleArray to an EnsembleScalar by applying np.asscalar
    
    Arguments:
       a (EnsembleScalar or EnsembleArray): ensemble variable to be turned to scalar
    """

    if isinstance(a, EnsembleScalar):
        return a
    else:
        data = OrderedDict()
        for block, aa in a:
            data[block] = np.asscalar(aa)
        return EnsembleScalar(a.ensemble, data)
