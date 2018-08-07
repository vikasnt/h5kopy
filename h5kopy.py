"""h5kopy module.

module to take multiple hdf5 files as input and append
them to produce one output file,based on user input condition(s)
for the KATRIN experiment data.
"""

import fnmatch
import logging
import configparser
import glob
from itertools import product
import h5py
import numpy as np
prmtr_names = []
spr = []
use = []
filelist = []


def dataset_copy(item, item_len, new, new_len):
    """Fuction to copy dataset from item into new."""
    # dataset exists, resize and modify

    if item.name in new:

        # Case 1: 1 data point dataset for mean values
        if len(item.shape) == 1 and item.len() == 1:
            # need to take weighted average here
            new[item.name][0] = (new[item.name][0]*new_len +
                                 item.value*item_len)/(new_len+item_len)

        # Case 2: 1d /2d array
        else:
            x_len = new[item.name].len()+item.len()
            new[item.name].resize(x_len, axis=0)
            new[item.name][-item.shape[0]:] = item.value

    # dataset doesn't exist, create dataset and modify
    # this might never be needed if we only update existing files
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
            # TODO: because of this line, code wont work with attrs of groups
            if len(item.shape) == 1 and item.len() == 1:
                # Error attribute for mean values added as sqrt(x^2+y^2)
                # Unit attribute is not appended
                if name == 'Error':
                    attr_data = np.add(new[item.name].attrs.get(name)**(2),
                                       item.attrs.get(name)**(2))**(1/2)
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
                                                     item.attrs.get(name))
                else:
                    new[item.name].attrs.__setitem__(name,
                                                     item.attrs.get(name))
            else:
                new[item.name].attrs.__setitem__(name, item.attrs.get(name))


# item =file to be merged, new = final file
# pass len argument as 0 when new is an empty file
def copy(item, item_len, new, new_len):
    """Fuction to copy input file data into output file."""
    # if item is group
    if isinstance(item, h5py.Group):
        new.require_group(item.name)

    # if item is dataset
    if isinstance(item, h5py.Dataset):
        dataset_copy(item, item_len, new, new_len)

    # copy item attribute(s)
    attr_copy(item, new)

    # continue towards sub-groups/datasets, if item is group
    if isinstance(item, h5py.Group):
        for name in item:
            copy(item[name], item_len, new, new_len)


def show(new):
    """Use this function to.

    print name of mean value datasets
    """
    if isinstance(new, h5py.Group):
        for name in new:
            show(new[name])

    if isinstance(new, h5py.Dataset):
        if len(new.shape) == 1 and new.len() == 1:
            logging.info(new.name)


def init():
    """Fuction to take input.

    To read one input file, print available datasets
    for comparison and to read input.cfg
    """
    # print name of available parameters:
    total_prmtr = 0
    for filename in glob.glob('*.h5'):
        if not fnmatch.fnmatch(filename, 'out*'):
            filelist.append(filename)
    # if no .h5 files found, pass -ve value to main code
    if not filelist:
        total_prmtr = -1
    else:
        first_file = h5py.File(filelist[0], "r")
        logging.info("Available datasets of mean values in input :")
        show(first_file)
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


def close(file1, file2):
    """Check if input files are close enough for merge."""
    for k, prmtr in enumerate(prmtr_names):
        val1 = float(np.mean(file1[prmtr]))
        val2 = float(np.mean(file2[prmtr]))
        if not abs(val1-val2) < max(val1*spr[k], val2*spr[k]):
            return False
    return True


def move(file1, file2, new_count):
    """Code to execute how files are copied when in merge range"""
    len1 = file1["RunSummary/Counts"].len()
    len2 = file2["RunSummary/Counts"].len()

    # file1 is an output file
    if 'Filecount' in file1.attrs:
        copy(file2, len2, file1, len1)
        file1.attrs.__setitem__('Filecount', file1.attrs.get('Filecount')+1)
        filelist.remove(file2.filename)
    # file2 is an output file
    elif 'Filecount' in file2.attrs:
        copy(file1, len1, file2, len2)
        file2.attrs.__setitem__('Filecount', file2.attrs.get('Filecount')+1)
        filelist.remove(file1.filename)
    # both are input files
    else:
        new_name = 'out'+str(new_count)+'.h5'
        # output files will be overwritten if exists
        newfile = h5py.File(new_name,'w')
        newfile.attrs.create('Filecount', 2, (1,), 'int64')
        new_count += 1
        copy(file1, len1, newfile, 0)
        copy(file2, len2, newfile, newfile["RunSummary/Counts"].len())
        filelist[filelist.index(file1.filename)] = new_name
        filelist.remove(file2.filename)
        newfile.close()
    return new_count


def group():
    """Group files based on input."""
    # try to add files using bottoms up
    # keep trying until no more merge can happen
    new_count = 1
    old_length = len(filelist)
    new_length = old_length - 1
    while new_length < old_length:
        old_length = len(filelist)
        for filename1, filename2 in product(filelist, filelist):
            if not filename1 == filename2:
                file1 = h5py.File(filename1)
                file2 = h5py.File(filename2)
                if close(file1, file2):
                    new_count = move(file1, file2, new_count)
                    # we modified list we were iterating over, break loop
                    break
                file1.close()
                file2.close()
        new_length = len(filelist)

    # at this point we might have some files which aren't merged with any other file
    # copy them into new output file, one for each
    for filename in filelist:
        if not fnmatch.fnmatch(filename, 'out*'):
            file1 = h5py.File(filename)
            len1 = file1["RunSummary/Counts"].len()
            new_name = 'out'+str(new_count)+'.h5'
            newfile = h5py.File(new_name,'w')
            new_count += 1
            newfile.attrs.create('Filecount', 1, (1,), 'int64')
            copy(file1, len1, newfile, 0)
            file1.close()
            newfile.close()
            filelist[filelist.index(filename)] = new_name
    print("final output files", filelist)


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

            # group files based on input
            group()

        else:
            logging.error("No parameter found in input.cfg, "
                          "will not merge any file")
