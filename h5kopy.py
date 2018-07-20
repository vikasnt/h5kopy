"""h5kopy module.

module to take multiple hdf5 files as input and append
them to produce one output file,based on user input condition(s)
for the KATRIN experiment data.
"""

import numpy as np
import logging
import configparser
import h5py
import glob
from itertools import product

prmtr_names = []
spr = []
prmtr_values = []
use = []
filelist = []

def dataset_copy(item, new):
    """Fuction to copy dataset from input files into output file."""
    # dataset exists, resize and modify
    if item.name in new:

        # Case 1: 1 data point dataset for mean values
        if len(item.shape) == 1 and item.len() == 1:
            # need to take mean later, will be done in mean()
            new[item.name][0] = new[item.name][0] + item.value

        # Case 2: 1d /2d array
        else:
            x_len = new[item.name].len()+item.len()
            new[item.name].resize(x_len, axis=0)
            new[item.name][-item.shape[0]:] = item.value

    # dataset doesn't exist, create dataset and modify
    else:

        # Case 1: 1 data point dataset like T2FractionMean
        if len(item.shape) == 1 and item.len() == 1:
            new.create_dataset(item.name, item.shape, item.dtype)
            new[item.name][0] = item.value

        # Case 2: 1d array
        if len(item.shape) == 1 and item.len() != 1:
            new.create_dataset(item.name, item.shape, item.dtype,
                               chunks=True, maxshape=(500,))
            new[item.name][-item.shape[0]:] = item.value

        # Case 3: 2d array
        if len(item.shape) == 2:
            new.create_dataset(item.name, item.shape, item.dtype,
                               maxshape=(None, None))
            new[item.name][-item.shape[0]:] = item.value


def attr_copy(item, new):
    """Fuction to copy attribute from input files into output file."""
    for name in item.attrs:

        # attr exists in output file, modify attr data
        if name in new[item.name].attrs:

            # different treatment for attributes of dataset like T2FractionMean
            if len(item.shape) == 1 and item.len() == 1:
                # Error attribute for mean values added as sqrt(x^2+y^2)
                # (sqrt taken after all files copied in mean() routine)
                # Unit attribute is not appended
                if name == 'Error':
                    attr_data = np.add(new[item.name].attrs.get(name),
                                       item.attrs.get(name)**(2))
                    new[item.name].attrs.__setitem__(name, attr_data)
            else:
                attr_data = np.append(new[item.name].attrs.get(name),
                                      item.attrs.get(name))
                new[item.name].attrs.__setitem__(name, attr_data)

        # create new.attr and add data
        else:
            if len(item.shape) == 1 and item.len() == 1:
                # Error attribute for mean values
                if name == 'Error':
                    new[item.name].attrs.__setitem__(name,
                                                     item.attrs.get(name)**(2))
                else:
                    new[item.name].attrs.__setitem__(name,
                                                     item.attrs.get(name))
            else:
                new[item.name].attrs.__setitem__(name, item.attrs.get(name))


# item =file to be merged, new = final file
def copy(item, new):
    """Fuction to copy input file data into output file."""
    # if item is group
    if isinstance(item, h5py.Group):
        new.require_group(item.name)

    # if item is dataset
    if isinstance(item, h5py.Dataset):
        dataset_copy(item, new)

    # copy item attribute(s)
    attr_copy(item, new)

    # continue towards sub-groups/datasets, if item is group
    if isinstance(item, h5py.Group):
        for name in item:
            copy(item[name], new)


def mean(new, value):
    """Use this function to.

    1) print name of mean value datasets ( when value=0 )
    2) get average value for mean value dataset
    3) take sqrt of error attribute for mean value dataset.
    """
    if isinstance(new, h5py.Group):
        for name in new:
            mean(new[name], value)

    if isinstance(new, h5py.Dataset):
        if len(new.shape) == 1 and new.len() == 1:
            if value == 0:
                logging.info(new.name)
            if value > 0:
                new[0] = new[0]/value

    if value > 0:
        for name in new.attrs:
            if len(new.shape) == 1 and new.len() == 1:
                # sqrt taken for Error attribute
                if name == 'Error':
                    attr_data = new.attrs.get(name)**(1/2)
                    new.attrs.__setitem__(name, attr_data)


def init():
    """Fuction to take input.

    To read one input file, print available datasets
    for comparison and to read input.cfg
    """
    # print name of available parameters:
    total_prmtr = 0
    input_files = glob.glob("*.h5")
    # if no .h5 files found, pass -ve value to main code
    if len(input_files) == 0:
        total_prmtr = -1
    else:
        first_file = h5py.File(input_files[0], "r")
        logging.info("Available datasets of mean values in input :")
        mean(first_file, 0)
        print("\n")
        first_file.close()

        # read input from input.config
        config = configparser.ConfigParser()
        config.read('input.cfg')
        for sections in config.sections():
            prmtr_names.append(config.get(sections, 'name'))
            spr.append(config.getfloat(sections, 'spread'))
            total_prmtr += 1
    return total_prmtr

def close(i,j):
    """Function to check if input files are close enough for merge"""
    for k in range(len(prmtr_names)):
        if not (abs(prmtr_values[k][i]-prmtr_values[k][j]) < max(prmtr_values[k][i]*spr[k],prmtr_values[k][j]*spr[k]) ):
            return False
    return True




def group():
    """Function to group files based on input"""
    print(filelist)
    print(prmtr_values)

    # try to add files using bottoms up
    # keep trying until no more merge can happen
    turn=0
    old_length = len(filelist)
    new_length = old_length - 1
    while(new_length < old_length):
        old_length=len(filelist)
        for i, j in product(range(len(filelist)),range(len(filelist))):
            turn +=1
            if not i == j:
                if close(i,j):
                    # found one pair that can be added
                    file1=h5py.File(filelist[i])
                    file2=h5py.File(filelist[j])
                    copy(file2,file1)
                    mean(file1,2)
                    del filelist[j]
                    for k in range(len(prmtr_names)):
                       prmtr_values[k][i]=float(np.mean(file1[prmtr_names[k]]))
                       del prmtr_values[k][j]
                    new_length = len(filelist)
                    # we modified list we were iterating over, break loop
                    break
        new_length = len(filelist)
    logging.info("number of steps %d",turn)
    print("final files list", filelist)




# set logger level ( if needed diff from default )
logging.basicConfig(level=logging.INFO)

# if running as script
if __name__ == "__main__":
    num = init()
    counter = 0
    # no input files found by init()
    if num < 0:
        logging.error("No input files found, will terminate script")
    else:
        if num > 0:
            # print input parameter list
            logging.info("Parameters found in input.cfg :")
            while counter < num:
                logging.info("Parameter %s Spread %s",
                             prmtr_names[counter], spr[counter])
                counter += 1
            print("\n")

            # save all parameter values of interest in 2d array.
            counter = 0
            while counter < num:
                var = []
                for filename in glob.glob('*.h5'):
                    with h5py.File(filename) as f:
                        if counter == 0:
                            filelist.append(filename)
                        if not filename == 'new.h5':
                            var.append(float(np.mean(f[prmtr_names[counter]])))
                counter += 1
                prmtr_values.append(var)

            # group files based on input 
            group()
        else:
            logging.error("No parameter found in input.cfg, "
                            "will not merge any file")
