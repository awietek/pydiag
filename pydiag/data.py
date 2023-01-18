import os
import re
import h5py 
from collections import OrderedDict

def read_h5_file(filename, tags=None):
    """ Read data for 
    
    Args:
        filenane (str): directory containing all data files
    Returns:
        OrderedDict   : dictionary containing all entries of the hdf5 file
    """
    data = dict()

    def read_node(node):
        name = node.name[1:]
        dtype = node[:].dtype

        # Parsing a complex number
        if len(dtype) == 2:
            rname = dtype.descr[0][0]
            iname = dtype.descr[1][0]
            rtype = dtype.descr[0][1]
            itype = dtype.descr[1][1]
            if (rname == "real" and iname == "imag") or \
               (rname == "r" and iname == "i"):

                if (rtype == itype):
                    return node[:][rname] + 1j*node[:][iname]   
                else:
                    raise TypeError("Malformed complex numbers: "
                                    "real and imaginary datatypes do not agree")
            else:
                raise TypeError("Malformed complex numbers: "
                                "real and imaginary part not properly named")
        # Plain data
        elif len(dtype) == 0:
            return node[:]
        else:
            raise TypeError("Invalid type of dataspace")
                
    # Read all tags in data
    if tags == None:
        def visitor(name, node):
            name = node.name[1:]
            if isinstance(node, h5py.Dataset):
                data[name] = read_node(node)
        with h5py.File(filename, 'r') as fl:
            fl.visititems(visitor)

    # only read specified tags
    else:
        if not all(isinstance(tag, str) for tag in tags):
            raise TypeError("tags must be a list of strings")

        with h5py.File(filename, 'r') as fl:
            for tag in tags:
                node = fl[tag]
                if isinstance(node, h5py.Dataset):
                    data[tag] = read_node(fl[tag])
                else:
                    raise ValueError("Invalid tag name: node \"{}\" not found in hdf5 file".format(tag))
        
    return OrderedDict(sorted(data.items()))
   

def read_h5_data(directory, regex, tags=None):
    """ Read data in a directory given as hdf5 files of a certain regular expression
        
    Args:
        directory (str)   : directory containing all data files
        regex (str)       : regular expression to match files in the directory
    Returns:
        OrderedDict : ordered dictionary, keys are tuples of parameters, values are
                      dictionaries containing the hdf5 data
    """
    data = dict()
    for (dirname, _, filenames) in os.walk(directory):
        for fl in filenames:
            match = re.search(regex, fl)
            if match:
                group = match.groups()
                filename = os.path.join(dirname, fl)
                data[group] = read_h5_file(filename, tags=tags)

    return OrderedDict(sorted(data.items()))


            
    
            
