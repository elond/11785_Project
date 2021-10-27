from data_convert import *
import os

if __name__ == "__main__":
    datasets = ["GenBank/","RefSeq/"]
    input_file_names = ["neg_unique_reads.fasta", "pos_unique_reads.fasta"]
    data_dir = "/home/esteban/Documents/School/Class_11785/Project/Data/"
    output_dir = data_dir + "NPY_Data/"

    # Checks to see if output directory exists and creates directory if non-existent
    if os.path.isdir(output_dir) != True:
        print("Creating directory at: " + output_dir)
        os.mkdir(output_dir)

    for dataset in datasets:
        for input_file_name in input_file_names:
            input_file = data_dir + dataset + input_file_name
            output_path = output_dir + dataset
            fasta_to_npy(input_file, output_path)
