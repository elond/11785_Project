"""
Summary: Creates reads of a certain length from a specific genetic sequence
"""
from Bio import SeqIO
import pandas as pd
import os
import sys
import random
import csv

READ_LENGTH = 250
NUM_OF_READS = 2e6
dataset = "RefSeq"
out_path = "/home/esteban/Documents/School/Class_11785/Project/Data/{}/".format(dataset)
data_csv = out_path + "read_info.csv"

def CreateExpCSV(filename, RandomSelection, OutputDirectory):
    an_list, gs_list, fold_list = [],[],[]
    print("Appending {} data to csv...".format(filename.split("/")[-1]))
    for seq_record in SeqIO.parse(filename, "fasta"):
        seq_len = len(seq_record)
        if seq_len < 250:
            continue
        gs_list.append(seq_len)
        an_list.append(seq_record.id)
        chance = random.random()
        if chance <= RandomSelection:
            fold_list.append("train")
        else:
            fold_list.append("val")

    label_list = [filename.split("/")[-1].split("_")[0]]*len(an_list)
    data = {"fold1": fold_list, "assembly_accession": an_list,
            "Genome.Size": gs_list, "Label": label_list}
    csv_data = pd.DataFrame(data)
    csv_data.to_csv(OutputDirectory + "read_info.csv", mode="a", index=False, header=False)

def ReadGeneration(fasta, read_len, out_path, label, filename,content,rpg):
    """
    Summary: Generates (read_len) bp long reads from input sequence
    Parameters:
        fasta: Input fasta file
    Returns:
        read_list: List of READ_LENGTH bp long reads
    """
    labeled_data = content[content["Label"] == label]
    for fold in ["train","val"]:
        fold_data = labeled_data[labeled_data["fold1"] == fold]
        seq_list = []
        seq_db = []
        for seq_record in SeqIO.parse(fasta, "fasta"):
            if seq_record.id in fold_data.assembly_accession.tolist():
                seq_list.append(str(seq_record.seq))

        print("Starting to generate reads for {}".format(fasta.split("/")[-1]))
        for sequence in seq_list:
            seq_length = len(sequence)-read_len
            temp_rpg = rpg
            if seq_length < rpg:
                temp_rpg = seq_length + 1
            read_list = [0]*temp_rpg
            pos = random.sample(range(seq_length+1),temp_rpg)
            for x in range(temp_rpg):
                read_list[x] = sequence[pos[x]:pos[x]+read_len]
            seq_db.append(list(set(read_list)))

        seq_db = list(set(sum(seq_db, [])))
        print("Finished generating {} reads for {}".format(len(seq_db),fasta.split("/")[-1]))
        ReadFolderCreation(seq_db, out_path, filename, label, fold)

def ReadFolderCreation(read_list, out_path, fasta_name, label, fold):
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
    org_name = fasta_name.split("_")[0]

    if fold == "train":
        out_file = out_path + "train_unique_reads.fasta"
    else:
        out_file = out_path + "val_unique_reads.fasta"

    print("Starting to organize reads for {}".format(org_name))
    with open(out_file, "a+") as file:
        for read in read_list:
            file.write(">" + org_name + "\n" + read + "\n")
    print("Finished organizing reads for {}".format(org_name))
    return

# Code starts here
if data_csv not in os.listdir(out_path):
    with open(data_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fold1","assembly_accession","Genome.Size","Label"])
    for filename in os.listdir(out_path):
        if filename.endswith(".fasta"):
            file_path = out_path+filename
            label = int(filename.split("_")[0])
            CreateExpCSV(file_path, 0.8, out_path)

# Determine reads per genome
content = pd.read_csv(out_path+"read_info.csv")
rpg = int(NUM_OF_READS//len(content.assembly_accession))

for filename in os.listdir(out_path):
    if filename.endswith(".fasta"):
        file_path = out_path+filename
        label = int(filename.split("_")[0])
        ReadGeneration(file_path, READ_LENGTH, out_path, label, filename, content, rpg)
