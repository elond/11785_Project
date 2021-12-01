"""
Summary: Creates reads of a certain length from a specific genetic sequence
"""
from Bio import SeqIO
import pandas as pd
import os
import sys
import random
import csv
import time

READ_LENGTH = 250
NUM_OF_READS = 1e6
dataset = "GenBank"
out_path = "/home/esteban/Documents/School/Class_11785/Project/Data/{}/".format(dataset)
data_csv = out_path + "read_info.csv"
CreateCSV = False

def CreateExpCSV(filename, RandomSelection, OutputDirectory):
    an_list, gs_list, mr_list, fold_list = [],[],[],[]
    print("Appending {} data to csv...".format(filename.split("/")[-1]))
    for seq_record in SeqIO.parse(filename, "fasta"):
        seq_len = len(seq_record)
        gs_list.append(seq_len)
        if seq_len < 250:
            mr_list.append(1)
        else:
            mr_list.append(seq_len-250+1)
        an_list.append(seq_record.id)
        chance = random.random()
        if chance <= RandomSelection:
            fold_list.append("train")
        else:
            fold_list.append("val")

    label_list = [filename.split("/")[-1].split("_")[0]]*len(an_list)
    data = {"fold1": fold_list, "assembly_accession": an_list,
            "Genome.Size": gs_list,"Max.Reads": mr_list,"Label": label_list}
    csv_data = pd.DataFrame(data)

    # Check to confirm each dataset has at least one genome
    check_list = ["train","val"]
    for val in csv_data["Label"].unique():
        temp = csv_data[csv_data["Label"] == val]
        check = [elem for elem in temp["fold1"].unique() if elem not in check_list]
        for thing in check:
            an = temp["assembly_accession"][0]
            csv_data.loc[csv_data.assembly_accession == an] = thing

    # Output
    csv_data.to_csv(OutputDirectory + "read_info.csv", mode="a", index=False, header=False)

def ReadGeneration(fasta, read_len, out_path, label, filename,content,rpc):
    """
    Summary: Generates (read_len) bp long reads from input sequence
    Parameters:
        fasta: Input fasta file
    Returns:
        read_list: List of READ_LENGTH bp long reads
    """
    fold_list = ["train","val"]
    labeled_data = content[content["Label"] == label]
    rpc_1 = rpc.loc[rpc["Label"] == label, "rpc"].values[0]
    fold_max_reads = [labeled_data.loc[labeled_data["fold1"] == fold, "Max.Reads"].sum() for fold in fold_list]
    percent_reads = fold_max_reads/sum(fold_max_reads)
    ex_sum = 0
    print("Starting to generate reads for {}".format(fasta.split("/")[-1]))
    for i,fold in enumerate(fold_list):
        start_time = time.time()
        reads = round(percent_reads[i]*rpc_1)
        fold_data = labeled_data[labeled_data["fold1"] == fold]
        seq_list = []
        seq_db = []
        print("Uploading sequences")
        for i,seq_record in enumerate(SeqIO.parse(fasta, "fasta"), start=1):
            if seq_record.id in fold_data.assembly_accession.tolist():
                seq_list.extend([str(seq_record.seq)])
            if i % 1000 == 0:
                print("Uploaded {0} genomes in {1:.2f} seconds".format(i, time.time() - start_time))

        print("Finished uploading {0} genomes in {1:.2f} seconds".format(len(seq_list), time.time()-start_time))
        rpg = round(reads/len(seq_list))

        print("Generating reads for {}".format(fasta.split("/")[-1]))
        new_start_time = time.time()
        for i,sequence in enumerate(seq_list, start=1):
            seq_len = len(sequence)
            seq_range = seq_len-read_len + 1
            temp_rpg = rpg
            ex = 0
            if seq_range < rpg:
                temp_rpg = seq_range
                ex = rpg - seq_range
            if temp_rpg < 1:
                temp_rpg = 1
                seq_range = seq_len
            if ex_sum > 0 and seq_range > rpg:
                if seq_range - rpg - ex_sum < 0:
                    diff = seq_range - rpg
                    temp_rpg += diff
                    ex_sum -= diff
                else:
                    temp_rpg += ex_sum
                    ex_sum = 0

            read_list = [0]*temp_rpg
            pos = random.sample(range(seq_range),temp_rpg)
            for x in range(temp_rpg):
                read_list[x] = sequence[pos[x]:pos[x]+read_len]
            seq_db.extend(list(set(read_list)))
            ex_sum += ex

            if i % 1000 == 0:
                print("Finished generating {0} unique sequences from {1} genomes in {2:.2f}".format(len(seq_db),i,time.time()-new_start_time))

        print("Determining whether reads are unique")
        seq_db = list(set(seq_db))
        print("Finished generating {0} reads for {1} in {2:.2f}".format(len(seq_db),fasta.split("/")[-1], time.time()-start_time))
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
if CreateCSV == True:
    with open(data_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fold1","assembly_accession","Genome.Size","Max.Reads","Label"])
    for filename in os.listdir(out_path):
        if filename.endswith(".fasta"):
            file_path = out_path+filename
            label = int(filename.split("_")[0])
            CreateExpCSV(file_path, 0.8, out_path)

# Determine reads per genome
content = pd.read_csv(out_path+"read_info.csv")
total_max_reads = content["Max.Reads"].sum()
label_max_reads = [content.loc[content["Label"] == thing, "Max.Reads"].sum() for thing in content["Label"].unique()]

standard_rpc = int(NUM_OF_READS//len(content["Label"].unique()))
excess = [mr - standard_rpc for mr in label_max_reads]
rpc = [standard_rpc]*len(content["Label"].unique())
run_sum = 0
pos_run_sum = 0
for i,ex in enumerate(excess):
    if ex < 0:
        run_sum += ex
        rpc[i] = label_max_reads[i]
    if ex > 0:
        pos_run_sum += ex
if -run_sum > pos_run_sum:
    maxi = pos_run_sum
elif -run_sum < pos_run_sum:
    maxi = -run_sum

temp_list = [0]*len(excess)
for i,ex in enumerate(excess):
    if ex > 0:
        temp_list[i] = ex
ex_reads = (temp_list/sum(temp_list))*maxi
ex_reads = [round(x) for x in ex_reads]
rpc = [a + b for a, b in zip(rpc, ex_reads)]
data = {"rpc": rpc, "Label": content["Label"].unique()}
rpc = pd.DataFrame(data)

for filename in os.listdir(out_path):
    if filename.endswith(".fasta"):
        file_path = out_path+filename
        label = int(filename.split("_")[0])
        ReadGeneration(file_path, READ_LENGTH, out_path, label, filename, content, rpc)
