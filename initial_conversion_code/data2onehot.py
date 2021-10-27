import numpy as np

# Function that converts a (N,L) shaped list into a one-hot encoding matrix
def one_hot_encode(read_list):
    """
    Summary: Converts a list of reads into their one-hot encoding representation
    Parameters:
        read_list: List of genetic reads
    Returns:
        seq_tensor: List of one-hot encoded reads
    """
    mapping = dict(zip("ACGT", range(4)))
    num_reads = len(read_list)
    seq_len = len(read_list[0])
    seq_tensor = []
    for seq in read_list:
        temp = np.zeros([seq_len, 4], dtype=int)
        for nuc in range(seq_len):
            if seq[nuc] in ["A","C","G","T"]:
                pos = mapping[seq[nuc]]
                temp[nuc, pos] = 1
        seq_tensor.append(temp)
    return np.array(seq_tensor)
