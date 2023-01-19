from .ensemble import Ensemble
from .array import Array
from .scalar import Scalar

import numpy as np
import scipy as sp

from collections import OrderedDict
    
def expm(A):
    """ Compute the matrix exponential of an ensemble.Array

    Args:
         A (ensemble.Array) : Input with last two dimensions are square (..., n, n).
    Returns:
         ensemble.Array : the resulting matrix exponential with the same shape of A
    """
    if isinstance(A, Array):
        data = OrderedDict()
        for block, arr in A.items():
            data[block] = sp.linalg.expm(arr)
        return Array(A.ensemble, data)
    else:
        raise TypeError("input needs to be an ensemble.Array")


def transpose(A):
    """ Returns an ensemble.Array with axes transposed.

    Args:
         A (ensemble.Array) : Input array
    Returns:
         ensemble.Array : A with its axes permuted.
    """
    if isinstance(A, Array):
        data = OrderedDict()
        for block, arr in A.items():
            data[block] = np.transpose(arr)
        return Array(A.ensemble, data)
    else:
        raise TypeError("input needs to be an ensemble.Array")


def conj(A):
    """ Returns complex conjugate of an ensemble.Array

    Args:
         A (ensebmle.Array) : Input array
    Returns:
         Array : complex conjugate of A
    """
    if isinstance(A, Array):
        data = OrderedDict()
        for block, arr in A.items():
            data[block] = np.conj(arr)
        return Array(A.ensemble, data)
    else:
        raise TypeError("input needs to be an ensemble.Array")


def dot(a, b):
    """ dot product of two ensemble.Arrays as defined by numpy.dot

    Args:
         a (ensemble.Array) : First array
         b (ensemble.Array) : First array
    Returns:
         ensemble.Array : dot product of all blocks
    """
    if isinstance(a, Array) and isinstance(b, Array):
        data = OrderedDict()
        if a.ensemble != b.ensemble:
            raise ValueError("a and b do not share the same ensemble")

        for block, deg in a.ensemble:
            data[block] = np.dot(a.array[block], b.array[block])
        return Array(a.ensemble, data)
    else:
        raise TypeError("inputs need to be ensemble.Array")

def outer(a, b):
    """ outer product of two ensemble.Arrays as defined by numpy.outer

    Args:
         a (ensemble.Array) : First array
         b (ensemble.Array) : First array
    Returns:
         ensemble.Array : outer product of all blocks
    """
    if isinstance(a, Array) and isinstance(b, Array):
        data = OrderedDict()
        if a.ensemble != b.ensemble:
            raise ValueError("a and b do not share the same ensemble")

        for block, deg in a.ensemble:
            data[block] = np.outer(a.array[block], b.array[block])
        return Array(a.ensemble, data)
    else:
        raise TypeError("inputs need to be ensemble.Array")

def add_outer(a, b):
    """ outer addition of two ensemble.Arrays as defined by numpy.add.outer

    Args:
         a (ensemble.Array) : First array
         b (ensemble.Array) : First array
    Returns:
         ensemble.Array : outer addition of all blocks
    """
    if isinstance(a, Array) and isinstance(b, Array):
        data = OrderedDict()
        if a.ensemble != b.ensemble:
            raise ValueError("a and b do not share the same ensemble")

        for block, deg in a.ensemble:
            data[block] = np.add.outer(a.array[block], b.array[block])
        return Array(a.ensemble, data)
    else:
        raise TypeError("inputs need to be ensemble.Array")

def subtract_outer(a, b):
    """ outer subtraction of two ensemble.Arrays as defined by numpy.subtract.outer

    Args:
         a (ensemble.Array) : First array
         b (ensemble.Array) : First array
    Returns:
         ensemble.Array : outer subtraction of all blocks
    """
    if isinstance(a, Array) and isinstance(b, Array):
        data = OrderedDict()
        if a.ensemble != b.ensemble:
            raise ValueError("a and b do not share the same ensemble")

        for block, deg in a.ensemble:
            data[block] = np.subtract.outer(a.array[block], b.array[block])
        return Array(a.ensemble, data)
    else:
        raise TypeError("inputs need to be ensemble.Array")

    
    
def einsum(subscripts, *operands, optimize=False):
    """ Einstein summation on each block according to the numpy.einsum

    Args:
         subscripts (str) : Specifies the subscripts for summation as
                            comma separated list of subscript labels. 
                            see numpy.einsum documentation
         operands (list of ensemble.Array): These are the arrays for the operation.
    Returns:
         ensemble.Array : The calculation based on the Einstein summation convention.
    """
    if len(operands) == 0:
        raise ValueError("need at least one operand")
    
    for o in operands:
        if not isinstance(o, Array):
            raise TypeError("operands need to be ensemble.Array")

    ensembles = [o.ensemble for o in operands]
    if len(set(ensembles)) > 1:
        raise ValueError("operands do not share the same ensemble")
    
    data = OrderedDict()
    for block, deg in ensembles[0]:
        arrs = [o.array[block] for o in operands]
        data[block] = np.einsum(subscripts, *arrs, optimize=optimize)
    return Array(ensembles[0], data)
