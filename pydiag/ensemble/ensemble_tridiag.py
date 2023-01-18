from .ensemble import Ensemble
from .ensemble_array import EnsembleArray
from .ensemble_scalar import EnsembleScalar

import numpy as np
import scipy as sp
import scipy.linalg

from collections import OrderedDict

class EnsembleTriDiag:
    """ Class defining an ensemble of symmetric tridiagonal matrices
    
    Attributes:
        ensemble (Ensemble)  :  ensemble defining blocks and degeneracies
        diag (OrderedDict)   :  dictionary of diagonal entries for each block
        offdiag (OrderedDict):  dictionary of off-diagonal entries for each block
    """

    def __init__(self, ensemble, data, diag_tag="diag", offdiag_tag="offdiag", dtype=None):
        """ Constructor of EnsembleTriDiag
    
        Arguments:
            ensemble (Ensemble): ensemble defining blocks and degeneracies
            data (OrderedDict) : dictionary of arrays whose keys are blocks
            diag_tag (str)     : tag specifying the key of diagonals in data[block]
            offdiag_tag (str)  : tag specifying the key of off-diagonals in data[block]
            dtype (np.dtype)   : (optional) datatype of the scalar values
        """
        self.ensemble = ensemble
        self.diag = OrderedDict()
        self.offdiag = OrderedDict()        

        if not isinstance(ensemble, Ensemble):
            raise TypeError("ensemble is not of type pydiag.Ensemble")
        if not isinstance(data, OrderedDict):
            raise TypeError("data is not of type OrderedDict")
        if not isinstance(diag_tag, str) or not isinstance(offdiag_tag, str):
            raise TypeError("diag_tag / offdiag_tag both need to be strings")
        
        ens_diag = EnsembleArray(ensemble, data, tag=diag_tag, dtype=dtype)
        ens_offdiag = EnsembleArray(ensemble, data, tag=offdiag_tag, dtype=dtype)        

        for block, deg in ensemble:
            d = ens_diag.array[block].flatten()
            od = ens_offdiag.array[block].flatten()

            if len(d) == len(od):
                self.diag[block] = d
                self.offdiag[block] = od[:-1]
            elif len(d) == len(od)+1:
                self.diag[block] = d
                self.offdiag[block] = od
            else:
                raise ValueError("Incompatible diag/offdiag dimensions in blockm {}: {}/{}".format(\
                                    block, len(d), len(od)))

    def __str__(self):
        s = ""
        for block, deg in self.ensemble:
            s += str(block) + ": dim=" + str(len(self.diag[block])) + "\n"
        return s

    def __iter__(self):
        return zip(self.diag.keys().__iter__(), \
                   self.diag.values().__iter__(), \
                   self.offdiag.values().__iter__())

    def __next__(self):
        return zip(self.diag.keys().__next__(), \
                   self.diag.values().__next__(), \
                   self.offdiag.values().__next__())


    
    def keys(self):
        return self.diag.keys().__iter__()

    def values(self):
        return zip(self.diag.values().__iter__(), \
                   self.offdiag.values().__iter__())

    def items(self):
        return self.__iter__()


    
    def eig0(self):
        """ Compute the smallest eigenvalue of the tridiagonal blocks
    
        Returns:
            EnsembleScalar:  the smallest eigenvalue of each block
        """
        data_e0 = OrderedDict()
        for block, deg in self.ensemble:
            if len(self.diag[block]) == 0:
                data_e0[block] = None
            elif len(self.diag[block]) == 1:
                data_e0[block] = self.diag[block][0]
            else:
                data_e0[block] = \
                    sp.linalg.eigvalsh_tridiagonal(self.diag[block], self.offdiag[block],\
                                            'i', select_range=(0,0), check_finite=False)[0]
        return EnsembleScalar(self.ensemble, data_e0)
            
    def eig(self):
        """ Compute the eigen decomposition of the tridiagonal blocks
    
        Returns:
            EnsembleArray:  1d EnsembleArray containing the eigenvalues
            EnsembleArray:  2d EnsembleArray containing the eigenvectors
        """
        data_evals = OrderedDict()
        data_evecs = OrderedDict()
        for block, deg in self.ensemble:
            if len(self.diag[block]) == 0:
                data_evals[block] = np.empty((0,))
                data_evecs[block] = np.empty((0,0))
            elif len(self.diag[block]) == 1:
                data_evals[block] = self.diag[block]
                data_evecs[block] = np.ones((1,1))
            else:
                data_evals[block], data_evecs[block] = \
                    sp.linalg.eigh_tridiagonal(self.diag[block], self.offdiag[block],
                                               check_finite=False, select='a',
                                               lapack_driver='stev')
        return EnsembleArray(self.ensemble, data_evals), EnsembleArray(self.ensemble, data_evecs)

    def eigvals(self):
        """ Compute the eigenvalues of the tridiagonal blocks
    
        Returns:
            EnsembleArray:  1d EnsembleArray containing the eigenvalues
        """
        data_evals = OrderedDict()
        for block, deg in self.ensemble:
            if len(self.diag[block]) == 0:
                data_evals[block] = np.empty((0,))
            elif len(self.diag[block]) == 1:
                data_evals[block] = self.diag[block]
            else:
                data_evals[block] = sp.linalg.eigvalsh_tridiagonal(
                    self.diag[block], self.offdiag[block],
                    check_finite=False, select='a')

        return EnsembleArray(self.ensemble, data_evals)

    def asarray(self):
        """ Transform the tridiagonal matrices to proper 2d arrays
    
        Returns:
            EnsembleArray:  2d EnsembleArray containing the matrices
        """
        data_mats = OrderedDict()
        for block, deg in self.ensemble:
            l = len(self.diag[block])
            if self.diag[block] is None:
                data_mats[block] = np.empty((0,0))
            else:
                data_mats[block] = np.diag(self.diag[block]) + \
                               np.diag(self.offdiag[block], k=1) + \
                               np.diag(self.offdiag[block], k=-1)
        return EnsembleArray(self.ensemble, data_mats)
