import gen_encodings as enc
from readin_fasta import *
import numpy as np
import tables

def fasta_to_npy(input_file, output_path):
    """
    Summary:
        Takes in an fasta input directory and converts the data into two NPY
        files containing an one-hot encoding and a codon-action tensor (CAT)
        encoding.
    Parameters:
        input_file: path to input file
        output_path: path to ouput files
    """
    _, sequence_list = readin_fasta(input_file)
    # Convert from a list of seqeunces
    fragments_onehot = enc.one_hot_encode(sequence_list)
    fragments_cat = enc.cat_encode(sequence_list)

    # Output npy files
    with open(output_path + "onehot_encoding.h5", 'w') as f:
        np.save(f, fragments_onehot)

    with open(output_path + "cat_encoding.h5", 'w') as f:
        np.save(f, fragments_onehot)
