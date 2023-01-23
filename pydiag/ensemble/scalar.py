from .ensemble import Ensemble

import numpy as np

from collections import OrderedDict

class Scalar:
    """ Class defining an ensemble of scalar values
    
    Attributes:
        ensemble (Ensemble) :  ensemble defining blocks and degeneracies
        scalar (OrderedDict):  dictionary of scalar values for each block
    """

    def __init__(self, ensemble, data, tag=None, dtype=None):
        """ Constructor of ensemble.Scalar
    
        Arguments:
            ensemble (Ensemble): ensemble defining blocks and degeneracies
            data (OrderedDict) : dictionary of scalars whose keys are blocks
            tag (str)          : (optional) tag specifying the key if data[block]
                                 is a dictionary and not an array
            dtype (np.dtype)   : (optional) datatype of the scalar values
        """
        self.ensemble = ensemble
        self.scalar = OrderedDict()

        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble is not of type pydiag.Ensemble")
        if not isinstance(data, OrderedDict) and not isinstance(data, dict):
            raise TypeError("data is not of dictionary type")

        for block, deg in ensemble:
            if tag == None:
                dt = data[block]
            else:
                if not isinstance(tag, str):
                    raise TypeError("tag needs to be a string")
                dt = data[block][tag]

            if dt is None:
                self.scalar[block] = None
            else:
                if dtype == None:
                    if isinstance(dt, np.ndarray):
                        self.scalar[block] = dt.item()
                    else:
                        self.scalar[block] = dt
                else:
                    if isinstance(dt, np.ndarray):
                        self.scalar[block] = dtype(dt.item())
                    else:
                        self.scalar[block] = dtype(dt)
    
    def __str__(self):
        s = ""
        for block, deg in self.ensemble:
            s += str(block) + ": " + str(self.scalar[block]) + "\n"
        return s

    def __iter__(self):
        return self.scalar.items().__iter__()

    def __next__(self):
        return self.scalar.items().__next__()

    def __add__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr + other 
            return Scalar(self.ensemble, data)
        else:
            for block, arr in self.items():
                data[block] = arr + other[block] 
            return Scalar(self.ensemble, data)
        
    def __sub__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr - other 
            return Scalar(self.ensemble, data)
        else:
            for block, arr in self.items():
                data[block] = arr - other[block] 
            return Scalar(self.ensemble, data)

    def __neg__(self):
        data = OrderedDict()
        for block, arr in self.items():
            data[block] = -arr
        return Array(self.ensemble, data)
        
    def __mul__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr * other 
            return Scalar(self.ensemble, data)
        else:
            return other.__mul__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        data = OrderedDict()
        if np.isscalar(other):
            for block, arr in self.items():
                data[block] = arr / other 
            return Scalar(self.ensemble, data)
        else:
            return other.__truediv__(self)

    def __getitem__(self, k):
        return self.scalar[k]
    
    def keys(self):
        return self.scalar.keys().__iter__()

    def values(self):
        return self.scalar.values().__iter__()

    def items(self):
        return self.scalar.items().__iter__()


    def min(self):
        """ compute minimal value of scalars across the blocks
    
        Returns:
            scalar:  minimal scalar
        """
        return min([s for s in self.scalar.values() if s is not None])

    def max(self):
        """ compute max value of scalars across the blocks
    
        Returns:
            scalar:  maximal scalar
        """
        return max([s for s in self.scalar.values() if s is not None])
