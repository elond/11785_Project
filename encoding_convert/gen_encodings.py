import numpy as np
import tables

def one_hot_encode(read_list):
    """
    Summary:
    Converts a list of reads into their one-hot encoding representation
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

def cat_encode_fragment(fragment,dna_codon_mapping):
    """
    Summary:
        Converts a single genetic fragment into its CAT-encoding
    Parameters:
        fragment: fragment of a genetic sequence
        dna_codon_mapping: mapping of a DNA sequence with its associated codon
    Returns:
        cat_encoded_fragment: CAT-encoded fragment of a genetic sequence
    """
    result = np.zeros((3, (len(fragment) // 3), 21))

    for i in range(len(fragment) - 2):

        # 'N' means the base could not be identified due to DNA sequence quality, ignore
        # Maybe should remove the N from the sequence before encoding (scrubbing)
        if 'N' in fragment[i:i+3]:
          continue
        codon = dna_codon_mapping[fragment[i:i+3]]

        # i % 3 selects which of the 3 start position fragments to set
        # i // 3 slides forward in each fragment after setting for each start position
        # codon indexes the ones-hot
        result[i % 3][i // 3][codon] = 1

    return cat_encoded_fragment

def cat_encode(sequence_list):
    """
    Summary:
        Converts a list of genetic sequences into its CAT-encoding
    Parameters:
        sequence_list: list of genetic sequences
    Returns:
        encoded_sequence_array: numpy array of CAT-encoded sequences
    """
    # Transform data from N batches of L-long fragments (N x L)
    # into N x 3 x (L // 3) x 21 matrix
    # (3 different start positions, 21 CAT combinations in a ones hot matrix)
    # Function to iterate over a fragment to build the (3 x (L // 3) x 21 matrix)
    # Enumerate amino acids for ones-hot index
    ALANINE = 0
    ASPARAGINE = 1
    ASPARTIC_ACID = 2
    ARGININE = 3
    CYSTEINE = 4
    GLUTAMIC_ACID = 5
    GLUTAMINE = 6
    GLYCINE = 7
    HISTIDINE = 8
    ISOLEUCINE = 9
    LEUCINE = 10
    LYSINE = 11
    METHIONINE = 12
    PHENYLALANINE = 13
    PROLINE = 14
    SERINE = 15
    THREONINE = 16
    TRYPTOPHAN = 17
    TYROSINE = 18
    VALINE = 19
    STOP = 20  # should always be last

    # Amino acids sharing codon with the stop codons
    # SELENOCYSTEINE = 20
    # PYRROLYSINE = 21

    # Dictionary mapping codon to amino acids
    dna_codon_mapping = {
        "GCT": ALANINE,
        "GCC": ALANINE,
        "GCA": ALANINE,
        "GCG": ALANINE,
        "AAT": ASPARAGINE,
        "AAC": ASPARAGINE,
        "GAT": ASPARTIC_ACID,
        "GAC": ASPARTIC_ACID,
        "CGT": ARGININE,
        "CGC": ARGININE,
        "CGA": ARGININE,
        "CGG": ARGININE,
        "AGA": ARGININE,
        "AGG": ARGININE,
        "TGT": CYSTEINE,
        "TGC": CYSTEINE,
        "GAA": GLUTAMIC_ACID,
        "GAG": GLUTAMIC_ACID,
        "CAA": GLUTAMINE,
        "CAG": GLUTAMINE,
        "GGT": GLYCINE,
        "GGC": GLYCINE,
        "GGA": GLYCINE,
        "GGG": GLYCINE,
        "CAT": HISTIDINE,
        "CAC": HISTIDINE,
        "ATT": ISOLEUCINE,
        "ATC": ISOLEUCINE,
        "ATA": ISOLEUCINE,
        "TTA": LEUCINE,
        "TTG": LEUCINE,  # can also be a start codon
        "CTT": LEUCINE,
        "CTC": LEUCINE,
        "CTA": LEUCINE,
        "CTG": LEUCINE,
        "AAA": LYSINE,
        "AAG": LYSINE,
        "ATG": METHIONINE,  # can also be a start codon
        "TTT": PHENYLALANINE,
        "TTC": PHENYLALANINE,
        "CCT": PROLINE,
        "CCC": PROLINE,
        "CCA": PROLINE,
        "CCG": PROLINE,
        "TCT": SERINE,
        "TCC": SERINE,
        "TCA": SERINE,
        "TCG": SERINE,
        "AGT": SERINE,
        "AGC": SERINE,
        "ACT": THREONINE,
        "ACC": THREONINE,
        "ACA": THREONINE,
        "ACG": THREONINE,
        "TGG": TRYPTOPHAN,
        "TAT": TYROSINE,
        "TAC": TYROSINE,
        "GTT": VALINE,
        "GTC": VALINE,
        "GTA": VALINE,
        "GTG": VALINE,  # can also be a start codon

        # Encoding stop codons together (rather than as independent amino acids)
        "TGA": STOP,
        "TAG": STOP,
        "TAA": STOP,
    }

    encoded_sequence_array = np.zeros((len(sequence_list), 3, len(sequence_list[0]) // 3, STOP + 1))

    for i, frag in enumerate(sequence_list):
        encoded_sequence_array[i] = cat_encode_fragment(frag,dna_codon_mapping)

    return encoded_sequence_array
