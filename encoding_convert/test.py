import gen_encodings as enc
from readin_fasta import *
import numpy as np
import os
import sys
import time


a = np.array([0,0,0,0])
b = np.array([1,1,1,1])
with open("testing_thing.npy", mode='ab') as f:
    np.save(f, a)
with open("testing_thing.npy", mode='ab') as f:
    np.save(f, b)

with open("testing_thing.npy", mode='rb') as f:
    fsz = os.fstat(f.fileno()).st_size
    out = np.load(f)
    while f.tell() < fsz:
        out = np.vstack((out, np.load(f)))
