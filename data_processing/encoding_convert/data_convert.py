import gen_encodings as enc
from readin_fasta import *
import numpy as np
import os
import sys
import time

def fasta_to_npy(input_file, output_path, batch_size, fragment_len):
    """
    Summary:
        Takes in an fasta input directory and converts the data into two NPY
        files containing an one-hot encoding and a codon-action tensor (CAT)
        encoding.
    Parameters:
        input_file: path to input file
        output_path: path to ouput files
    """
    fragment_gntr = readin_fasta(input_file, batch_size)
    fold = input_file.split("/")[-1].split("_")[0]

    # Check whether the output directories exist
    if not(os.path.exists(output_path)):
        print("Creating directory at: " + output_path)
        os.mkdir(output_path)
    if not(os.path.exists(output_path+"One_hot/")):
        print("Creating directory at: " + output_path+"One_hot/")
        os.mkdir(output_path+"One_hot/")
    if not(os.path.exists(output_path+"CAT/")):
        print("Creating directory at: " + output_path+"CAT/")
        os.mkdir(output_path+"CAT/")

    start_time = time.time()
    # Convert from a list of seqeunces
    for i,data in enumerate(fragment_gntr, 1):
        title_list, fragment_list = data
        fragments_onehot = enc.one_hot_encode(fragment_list, fragment_len)
        fragments_cat = enc.cat_encode(fragment_list, fragment_len)

        output_npy_files(output_path, fragments_onehot, fragments_cat, title_list, fold)

def output_npy_files(output_path, fragments_onehot, fragments_cat,title_list,dataset):
    with open(output_path + "One_hot/{}_data.npy".format(dataset), mode='ab') as f:
        np.save(f, fragments_onehot)
    with open(output_path + "One_hot/{}_labels.npy".format(dataset), mode='ab') as f:
        np.save(f, title_list)

    with open(output_path + "CAT/{}_data.npy".format(dataset), 'ab') as f:
        np.save(f, fragments_cat)
    with open(output_path + "CAT/{}_labels.npy".format(dataset), 'ab') as f:
        np.save(f, title_list)
