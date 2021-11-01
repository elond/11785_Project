import gen_encodings as enc
from readin_fasta import *
import numpy as np
import os
import sys
import time

input_file = 
fragment_gntr = readin_fasta(input_file, batch_size)

for i,data in enumerate(fragment_gntr, 1):
    title_list, fragment_list = data
    if i%8 == 0:
