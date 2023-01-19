from .ensemble import Ensemble

import numpy as np

from collections import OrderedDict

class Array:
    """ Class defining an ensemble of array values
    
    Attributes:
        ensemble (Ensemble) :  ensemble defining blocks and degeneracies
        array (OrderedDict) :  dictionary of scalar values for each block
        ndim  (int)         :  number of dimensions of the arrays
    """

    def __init__(self, ensemble, data, tag=None, dtype=None):
        """ Constructor of ensemble.Array
    
        Arguments:
            ensemble (Ensemble): ensemble defining blocks and degeneracies
            data (OrderedDict) : dictionary of arrays whose keys are blocks
            tag (str)          : (optional) tag specifying the key if data[block]
                                 is a dictionary and not an array
            dtype (np.dtype)   : (optional) datatype of the scalar values
        """
        self.ensemble = ensemble
        self.array = OrderedDict()
        self.ndim = 1

        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble is not of type pydiag.Ensemble")
        if not isinstance(data, OrderedDict):
            raise TypeError("data is not of type OrderedDict")

        if tag == None:
            for block, deg in ensemble:
                self.array[block] = np.array(data[block], dtype=dtype)
        else:
            if not isinstance(tag, str):
                raise TypeError("tag needs to be a string")

            for block, deg in ensemble:
                self.array[block] = np.array(data[block][tag], dtype=dtype)

        # determine number of dimensions
        ndims = np.unique([ar.ndim for ar in self.array.values()])
        if len(ndims) != 1:
            raise ValueError("not all arrays have same number of dimensions")
        else:
            self.ndim = ndims[0]
        
    def __str__(self):
        s = ""
        for block, deg in self.ensemble:
            s += str(block) + ": shape=" + str(self.array[block].shape) + "\n"
        return s

    def __iter__(self):
        return self.array.items().__iter__()

    def __next__(self):
        return self.array.items().__next__()

    def __add__(self, other):

        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr + other 
            return Array(self.ensemble, data)
        else:
            for block, arr in self.items():
                data[block] = arr + other[block] 
            return Array(self.ensemble, data)
        
    def __sub__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr - other 
            return Array(self.ensemble, data)
        else:
            for block, arr in self.items():
                data[block] = arr - other[block] 
            return Array(self.ensemble, data)

    def __mul__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr * other 
        else:
            for block, arr in self.items():
                data[block] = arr * other[block] 
        return Array(self.ensemble, data)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr / other
        else:
            for block, arr in self.items():
                data[block] = arr / other[block]
        return Array(self.ensemble, data)

    def __getitem__(self, k):
        data = OrderedDict()
        for block, arr in self.items():
            if len(arr) > 0:
                data[block] = arr[k]
            else:
                data[block] = np.empty(self.ndim * (0))
        return Array(self.ensemble, data)
             
    def keys(self):
        return self.array.keys().__iter__()

    def values(self):
        return self.array.values().__iter__()

    def items(self):
        return self.array.items().__iter__()

    def flatten(self):
        """
        Returns a flattened version of the entries same as in numpy.flatten

        Returns:
            ensemble.Array: The flattened array
        """
        data = OrderedDict()
        for block, arr in self.items():
            data[block] = arr.flatten()
        return Array(self.ensemble, data)
