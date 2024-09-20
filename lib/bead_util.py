import sys
import os
import fnmatch
import re
import numpy as np

def progress_bar(count, total, suffix='', bar_len=50, newline=True):
    '''Prints a progress bar and current completion percentage.
       This is useful when processing many files and ensuring
       a script is actually running and going through each file

           INPUTS: count, current counting index
                   total, total number of iterations to complete
                   suffix, option string to add to progress bar
                   bar_len, length of the progress bar in the console

           OUTPUTS: none
    '''
    
    if len(suffix):
        max_bar_len = 80 - len(suffix) - 17
        if bar_len > max_bar_len:
            bar_len = max_bar_len

    if count == total - 1:
        percents = 100.0
        bar = '#' * bar_len
    else:
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
    
    # This next bit writes the current progress bar to stdout, changing
    # the string slightly depending on the value of percents (1, 2 or 3 digits), 
    # so the final length of the displayed string stays constant.
    if count == total - 1:
        sys.stdout.write('[%s] %s%s ... %s\r' % (bar, percents, '%', suffix))
    else:
        if percents < 10:
            sys.stdout.write('[%s]   %s%s ... %s\r' % (bar, percents, '%', suffix))
        else:
            sys.stdout.write('[%s]  %s%s ... %s\r' % (bar, percents, '%', suffix))

    sys.stdout.flush()
    
    if (count == total - 1) and newline:
        print()

def make_all_pardirs(path):
    '''Function to help pickle from being shit. Takes a path
       and looks at all the parent directories etc and tries 
       making them if they don't exist.

       INPUTS: path, any path which needs a hierarchy already 
                     in the file system before being used

       OUTPUTS: none
       '''

    parts = path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)

def find_all_fnames(dirlist, ext='.h5', exclude_fpga=True, \
                    verbose=True, substr='', subdir='', \
                    use_origin_timestamp=False, \
                    skip_subdirectories=False):
    '''Finds all the filenames matching a particular extension
       type in the directory and its subdirectories .

       INPUTS: 
            dirlist, list of directory names to loop over

            ext, file extension you're looking for
            
            sort, boolean specifying whether to do a simple sort
            
            exclude_fpga, boolean to ignore hdf5 files directly from
                the fpga given that the classes load these automatically
               
            verbose, print shit
            
            substr, string within the the child filename to match
            
            subdir, string within the full parent director to match
               
            sort_time, sort files by timestamp, trying to use the hdf5 
                timestamp first

            use_origin_timestamp, boolean to tell the time sorting to
                use the file creation timestamp itself

            skip_subdirectories, boolean to skip recursion into child
                directories when data is organized poorly

       OUTPUTS: 
            files, list of filenames as strings, or list of lists if 
                multiple input directories were given

            lengths, length of file lists found '''

    if verbose:
        print("Finding files in: ")
        print(dirlist)
        sys.stdout.flush()

    was_list = True

    lengths = []
    files = []

    if type(dirlist) == str:
        dirlist = [dirlist]
        was_list = False

    for dirname in dirlist:
        for root, dirnames, filenames in os.walk(dirname):
            if (root != dirname) and skip_subdirectories:
                continue
            for filename in fnmatch.filter(filenames, '*' + ext):
                if ('_fpga.h5' in filename) and exclude_fpga:
                    continue
                if substr and (substr not in filename):
                    continue
                if subdir and (subdir not in root):
                    continue
                files.append(os.path.join(root, filename))
        if was_list:
            if len(lengths) == 0:
                lengths.append(len(files))
            else:
                lengths.append(len(files) - np.sum(lengths)) 

    if len(files) == 0:
        print("DIDN'T FIND ANY FILES :(")

    if 'new_trap' in files[0]:
        new_trap = True
    else:
        new_trap = False

    if verbose:
        print("Found %i files..." % len(files))
    if was_list:
        return files, lengths
    else:
        return files, [len(files)]