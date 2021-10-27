"""
Summary: Creates reads of a certain length from a specific genetic sequence
"""
from Bio import SeqIO
import pandas as pd
import os
import sys
import random

READ_LENGTH = 250
data_size_limit = 10000000
dataset = "GenBank"
out_path = "/home/esteban/Documents/School/Class_11785/Project/Data/{}/".format(dataset)

def ReadGeneration(fasta, read_len, data_size_limit):
    """
    Summary: Generates (read_len) bp long reads from input sequence
    Parameters:
        fasta: Input fasta file
    Returns:
        read_list: List of READ_LENGTH bp long reads
    """
    seq_list = []
    seq_db = []
    for seq_record in SeqIO.parse(fasta, "fasta"):
        seq_list.append(str(seq_record.seq))

    # randomly select from sequence list
    sampling = os.path.getsize(fasta) > data_size_limit
    if sampling == True:
        samp_num = round((len(seq_list)/os.path.getsize(fasta))*data_size_limit)
        seq_list = random.sample(seq_list,samp_num)

    print("Starting to generate reads for {}".format(fasta.split("/")[-1]))
    for sequence in seq_list:
        seq_length = len(sequence)
        read_list = [0]*(seq_length - read_len + 1)
        for x in range(seq_length - read_len + 1):
            read_list[x] = sequence[x:x+read_len]
        seq_db.append(list(set(read_list)))

    seq_db = list(set(sum(seq_db, [])))
    print("Finished generating {} reads for {}".format(len(seq_db),fasta.split("/")[-1]))
    return seq_db

def ReadFolderCreation(read_list, out_path, fasta_name, label):
    """
    Summary: Creates read folder with .csv file containing read labels
    Parameters:
        read_list: List of READ_LENGTH bp long reads
        fasta_name: List of name of species in FASTA file
        out_path: Path to output file
        label: Label of sequences
    """
    # Separate reads into folders
    total = len(read_list)
    org_name = fasta_name.split("_")[-1].split(".")[0]

    if fasta_name == "0_neg_data.fasta":
        out_file = out_path + "neg_unique_reads.fasta"
    else:
        out_file = out_path + "pos_unique_reads.fasta"

    print("Starting to organize reads for {}".format(org_name)
    with open(out_file, "a+") as file:
        for read in read_list:
            file.write(">" + org_name + "\n" + read + "\n")
    print("Finished organizing reads for {}".format(org_name))
    return

# Code starts here
for filename in os.listdir(out_path):
    if filename.endswith(".fasta"):
        label = int(filename.split("_")[0])
        read_list = ReadGeneration(out_path+filename, READ_LENGTH, data_size_limit)
        ReadFolderCreation(read_list, out_path, filename, label)
