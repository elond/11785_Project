"""
Summary: Downloads a list of genomes from NCBI following the search criteria.
Parameters:
    FILE_PATH: The path to the FASTA file
    DATABASE: NCBI Database to search
    RETMAX: The maximum amount of returned ans
    RETTYPE: The file type to return
    RETMODE: The file mode to return
"""
import os
from Bio import SeqIO, Entrez
import pandas as pd
import numpy as np
import sys

# Python function inputs
DATABASE = "nucleotide"
options = ["1_Adenovirus", "2_Astrovirus","3_Calicivirus","4_Enterovirus",
            "5_Hepatitis_A","6_Hepatitis_E","7_Rotavirus","8_Orthoreovirus"]
DATASET = ""
RETTYPE = "fasta"
RETMODE = "text"
INPUT_AN = '/home/esteban/Documents/School/Class_11785/Project/Data/an_list.csv'
FILE_PATH = "/home/esteban/Documents/School/Class_11785/Project/Data/"

# Necessary Entrez personal information
Entrez.email = "elondono@andrew.cmu.edu"
Entrez.api_key = "d861f4015ce9ead3a1b2aaa2442092b76d08"

# Code to download FASTA files
def DownloadFiles(an_list, database, return_type, return_mode, file_path):
    """
    Summary: Downloads the files of a list of ans
    Parameters:
        an_list: List of ans to be downloaded
        database: NCBI database to be downloaded from
        return_type: The file type to be downloaded
        return_mode: The file mode to be downloaded
        file_path: File path where data will be downloaded
    """
    if not os.path.isfile(file_path):
        # Downloads the an list according to Entrez standards (maximum 100,000 at a time)
        out_handle = open(file_path, "a+")
        total_seq = len(an_list)
        print("Downloading {} sequences".format(total_seq))

        if total_seq%1000 == 0:
            remaining = 0
        else:
            remaining = 1

        for x in range(total_seq//1000 + remaining):
            start = x*1000
            net_handle = Entrez.efetch(db=database,id=an_list,rettype=return_type,retmode=return_mode,retmax=1000,retstart=start)
            out_handle.write(net_handle.read())
            net_handle.close()
            print("Finished fetching {} files out of {}".format((x+1)*1000, total_seq))

        out_handle.close()
        print("Saved {} files to {}".format(return_type, file_path))
    else:
        print("Error: {} already exists".format(file_path))
        sys.exit()

###--------------------------------------------------------------------------###
# Start of Code
###--------------------------------------------------------------------------###
df = pd.read_csv(INPUT_AN)
neg = df.loc[:, df.columns != 'AN']
dataset_list = neg.columns.values.tolist()
data = df.loc[(neg==0).all(axis=1)]
for dataset in dataset_list:
    POS_PATH = FILE_PATH + dataset + ".fasta"
    ds = df.loc[df[dataset]==1]
    AN_LIST = ds["AN"].tolist()
    AN_LIST = list(map(str,AN_LIST))
    DownloadFiles(AN_LIST, DATABASE, RETTYPE, RETMODE, POS_PATH)
AN_LIST = data["AN"].tolist()
AN_LIST = list(map(str,AN_LIST))
DownloadFiles(AN_LIST, DATABASE, RETTYPE, RETMODE, FILE_PATH + "0_neg_data.fasta")
