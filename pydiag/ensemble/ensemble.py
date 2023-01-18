"""
Ensemble class

:author: Alexander Wietek
"""
import itertools
from collections import OrderedDict

class Ensemble:
    """ class defining an ensemble of blocks with degeneracy

    Example:
    ::
        $ nups = [(nup, 2) if nup != n_sites // 2 else (nup, 1) for nup in range(n_sites//2+1)]
        $ qs = [(q, 1) if q == 0 or q == n_sites // 2 else (q, 2) for q in range(n_sites//2+1)]
        $ ensemble = yd.Ensemble(nups, qs)
        $ for block, deg in ensemble:
        $     print("blocks", block, deg)

    Attributes:
        blocks (list of string tuples):  list of labels for the different blocks
        degeneracy (OrderedDict)      :  dictionary of degeneracies for a block
    """
    
    def __init__(self, *args, default_degeneracy=1):

        self.blocks = []
        self.degeneracy = OrderedDict()

        # convert each arg to degeneracy format
        args_formatted = []
        for arg in args:

            # get arg into form tuple(str, int)
            try:
                iterator = iter(arg)
            except TypeError:
                raise TypeError("block specifier not iterable")
            else:
                arg_formatted = []
                for obj in iterator:
                    # tuple specifying (block, degeneracy)
                    if isinstance(obj, tuple):
                        if len(obj) != 2:
                            raise ValueError("Invalid tuple length specifying (block, degeneracy)")
                        try:
                            arg_formatted.append((str(obj[0]), int(obj[1])))
                        except:
                            raise TypeError("Invalid tuple types specifying (block, degeneracy)")
                                                    
                    else:
                        try:
                            arg_formatted.append((str(obj), default_degeneracy))
                        except:
                            raise TypeError("Invalid type specifying block")
            args_formatted.append(arg_formatted)
            
        for blocks_degeneracies in itertools.product(*args_formatted):
            block = []
            deg = 1            
            for b, d in blocks_degeneracies:
                block.append(b)
                deg *= d
            block = tuple(block)
            
            self.blocks.append(block)
            self.degeneracy[block] = deg

        self.blocks = sorted(self.blocks)

    def __iter__(self):
        return self.degeneracy.items().__iter__()

    def __next__(self):
        return self.degeneracy.items().__next__()


    
