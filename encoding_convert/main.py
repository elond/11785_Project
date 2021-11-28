from data_convert import *
import os

if __name__ == "__main__":
    datasets = ["GenBank/", "RefSeq/"]
    input_file_names = ["train_unique_reads.fasta", "val_unique_reads.fasta"]
    data_dir = "/home/esteban/Documents/School/Class_11785/Project/Data/"
    output_dir = data_dir + "NPY_Data/"
    batch_size = 10000

    # Checks to see if output directory exists and creates directory if non-existent
    if not(os.path.isdir(output_dir)):
        print("Creating directory at: " + output_dir)
        os.mkdir(output_dir)

    for dataset in datasets:
        for input_file_name in input_file_names:
            input_file = data_dir + dataset + input_file_name
            output_path = output_dir + dataset
            print("Converting {}".format(input_file))
            fasta_to_npy(input_file, output_path, batch_size)
