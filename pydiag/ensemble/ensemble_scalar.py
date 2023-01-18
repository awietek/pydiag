from .ensemble import Ensemble

import numpy as np

from collections import OrderedDict

class EnsembleScalar:
    """ Class defining an ensemble of scalar values
    
    Attributes:
        ensemble (Ensemble) :  ensemble defining blocks and degeneracies
        scalar (OrderedDict):  dictionary of scalar values for each block
    """

    def __init__(self, ensemble, data, tag=None, dtype=None):
        """ Constructor of EnsembleScalar
    
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

        if tag == None:
            for block, deg in ensemble:
                dt = data[block]
                
                if dtype == None:
                    self.scalar[block] = dt
                else:
                    self.scalar[block] = dtype(dt)
        else:
            if not isinstance(tag, str):
                raise TypeError("tag needs to be a string")

            for block, deg in ensemble:
                dt = data[block]

                if dtype == None:
                    self.scalar[block] = dt
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

    
    def keys(self):
        return self.scalar.keys().__iter__()

    def values(self):
        return self.scalar.values().__iter__()

    def items(self):
        return self.scalar.items().__iter__()


    def min(self):
        """ compute minimal value of scalars
    
        Returns:
            scalar:  minimal scalar
        """
        return min([s for s in self.scalar.values() if s is not None])

    def max(self):
        """ compute max value of scalars
    
        Returns:
            scalar:  maximal scalar
        """
        return max([s for s in self.scalar.values() if s is not None])
