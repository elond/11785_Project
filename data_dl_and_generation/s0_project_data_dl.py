from Bio import SeqIO, Entrez
import pandas as pd
import numpy as np
import time
import sys
import os

# Necessary Entrez personal information
Entrez.email = "elondono@andrew.cmu.edu"
Entrez.api_key = "d861f4015ce9ead3a1b2aaa2442092b76d08"

# Inputs
Do_CreateCSV = True
Do_ExcludeData = False
Dataset = "Project"
OUT_PATH = "/home/esteban/Documents/School/Class_11785/Project/Data"
ExcludedData = OUT_PATH + "/pos_data.fasta"
OUT = OUT_PATH + "/an_list.csv"
DATABASE = "nucleotide"
REFER = "refseq" #or "genbank"
TERM = "complete genome[All Fields] AND (viruses[filter] AND biomol_genomic[PROP] AND {}[filter] AND is_nuccore[filter])".format(REFER)
TERM_DICT = {"1_Adenovirus": "AND Adenovirus[All Fields]",
            "2_Astrovirus": "AND Astrovirus[All Fields]",
            "3_Calicivirus": "AND Calicivirus[All Fields]",
            "4_Enterovirus": "AND Enterovirus[All Fields]",
            "5_Hepatitis_A": "AND (Hepatitis A[All Fields] OR Hepatovirus A[All Fields])",
            "6_Hepatitis_E": "AND Hepatitis_E[All Fields]",
            "7_Rotavirus": "AND Rotavirus[All Fields]",
            "8_Orthoreovirus": "AND Orthoreovirus[All Fields]"}

def DownloadANList(database, search_term):
    """
    Summary: Returns a list of accession numbers.
    Parameters:
        database: The NCBI database that will be searched
        search_term: The terms to be searched for in database
        return_max: The maximum amount of ANs to return
    Returns:
        an_list: A list of ans corresponding to the parameters
    """
    # Temporary search to find total amount of possible matches
    temp_search = Entrez.read(Entrez.esearch(db=database, term=search_term))
    total_seq = int(temp_search['Count'])
    return_max = 1000

    # Downloads the an list according to Entrez standards (100,000 at a time)
    an_list = []
    limit = return_max
    if return_max < total_seq:
        limit = total_seq
    if limit > 1000:
        remaining = 1
        if limit%1000 == 0:
            remaining = 0
        for x in range(limit//1000 + remaining):
            data = Entrez.esearch(db=database, retstart=x*1000, retmax=1000, term=search_term, idtype="acc")
            an_list = an_list + Entrez.read(data)['IdList']
            data.close()
            print("Downloaded {} genome ANs".format(len(an_list)))
            time.sleep(2)
    else:
        data = Entrez.esearch(db=database, retmax=return_max, term=search_term, idtype="acc")
        an_list = Entrez.read(data)['IdList']
        data.close()
    print("Downloaded {} ans from {} database".format(len(an_list), database))

    return an_list

def ExcludeData(df, path, e_data):
    #Extract ANs from negative dataset
    df["AN"].tolist()

    #Extract ANs from positive dataset
    pos_an_list = []
    pos_data = e_data
    for seq_record in SeqIO.parse(pos_data, "fasta"):
        pos_an_list.append(seq_record.id)

    #Remove positive ANs from Negative dataset
    dup_list = df.index[~~df["AN"].isin(pos_an_list)].tolist()
    df = df.drop(dup_list)

    # Download cleaned csv
    print("Number of dropped data: " + str(len(dup_list)))
    path = path + "/cleaned_neg_an_list.csv"
    df.to_csv(path, index=False)

###--------------------------------------------------------------------------###
# Start of code
###--------------------------------------------------------------------------###
if Do_CreateCSV == True:
    TERM_LIST = list(TERM_DICT.keys())
    FEATURE_LIST = ["AN"] + TERM_LIST
    AN_LIST = DownloadANList(DATABASE,TERM)
    df = pd.DataFrame(0, index=np.arange(len(AN_LIST)), columns=FEATURE_LIST)
    df["AN"] = AN_LIST
    # Updates dataset
    for r in TERM_LIST:
        NEW_TERM = TERM + " " + TERM_DICT[r]
        AN_LIST = DownloadANList(DATABASE,NEW_TERM)
        for an in AN_LIST:
            if an in df["AN"].tolist():
                i = df.index[df["AN"] == an]
                df[r][i] = 1
    df.to_csv(OUT, index=False)
if Do_ExcludeData == True:
    df = pd.read_csv(OUT_PATH + '/neg_an_list.csv')
    ExcludeData(df, OUT_PATH, ExcludedData)
