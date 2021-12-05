# Code to analyze coronaviruses
from Bio import SeqIO, Entrez
import numpy as np
import pandas as pd
import random

# Necessary Entrez personal information
Entrez.email = "elondono@andrew.cmu.edu"
Entrez.api_key = "d861f4015ce9ead3a1b2aaa2442092b76d08"

# Python function inputs
dataset = "RefSeq"
FolderPath = "/home/esteban/Documents/School/Class_11785/Project/Data/{}/".format(dataset)
RandomSelection = True
OutputDirectory = FolderPath

def CreateExpCSV(folder_path, RandomSelection, OutputDirectory):
    # Download negative csv_data and select AN column and label of negative
    neg_an_list = pd.read_csv(folder_path + "neg_data_an.csv")["AN"].tolist()
    neg_len = len(neg_an_list)
    neg_label_list = ["negative"]*neg_len
    neg_train_len = int(0.8*neg_len)
    neg_test_len = int((neg_len - neg_train_len)/2)
    neg_fold_list = ["train"]*neg_train_len + ["val"]*(neg_len-neg_train_len-neg_test_len) + ["test"]*(neg_test_len)
    if RandomSelection == True:
        random.shuffle(neg_fold_list)

    # Download positive csv_data and select AN column and label of positive
    pos_an_list = pd.read_csv(folder_path + "pos_data_an.csv")["AN"].tolist()
    pos_len = len(pos_an_list)
    pos_label_list = ["positive"]*len(pos_an_list)
    pos_train_len = int(0.8*pos_len)
    pos_test_len = int((pos_len - pos_train_len)/2)
    pos_fold_list = ["train"]*pos_train_len + ["val"]*(pos_len-pos_train_len-pos_test_len) + ["test"]*(pos_test_len)
    if RandomSelection == True:
        random.shuffle(pos_fold_list)

    # Combine lists
    an_list = neg_an_list + pos_an_list
    label_list = neg_label_list + pos_label_list
    fold_list = neg_fold_list + pos_fold_list

    # Create genome length list
    neg_data = list(SeqIO.parse(folder_path + "neg_data.fasta", "fasta"))
    pos_data = list(SeqIO.parse(folder_path + "pos_data.fasta", "fasta"))

    neg_gen_list = IndividualFASTA(neg_data, neg_fold_list, OutputDirectory)
    pos_gen_list = IndividualFASTA(pos_data, pos_fold_list, OutputDirectory)
    gen_list = neg_gen_list + pos_gen_list

    data = {"fold1": fold_list, "assembly_accession": an_list, "Genome.Size": gen_list, "Label": label_list}
    csv_data = pd.DataFrame(data)
    np.savetxt(folder_path + "pIMG_run.csv", csv_data, fmt = "%s", delimiter=",")

def IndividualFASTA(data, fold_list,ou_dir):
    gen_list = [0]*len(data)
    for i, x in enumerate(data):
        if "test" in fold_list[i]:
            output_handle = open(OutputDirectory + "Test_fasta/"+str(x.id+".fasta"), "w")
            SeqIO.write(x, output_handle, "fasta")
            gen_list[i] = len(x.seq)
            output_handle.close()
        else:
            output_handle = open(OutputDirectory + "Train_fasta/"+str(x.id+".fasta"), "w")
            SeqIO.write(x, output_handle, "fasta")
            gen_list[i] = len(x.seq)
            output_handle.close()
    return gen_list

# Extract fasta data
CreateExpCSV(FolderPath, RandomSelection, OutputDirectory)
