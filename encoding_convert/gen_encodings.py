import numpy as np
import tables
import sys

def one_hot_encode(fragment_list):
    """
    Summary:
        Converts a fragment into its one-hot encoding representation
    Parameters:
        fragment_list: A list of genetic fragments
    Returns:
        one_hot_list: A numpy array of one-hot encoded fragments
    """

    mapping = dict(zip("ACGT", range(4)))
    one_hot_list = []
    fragment_len = len(fragment_list[0])
    for fragment in fragment_list:
        fragment_onehot = np.zeros([fragment_len, 4], dtype="uint8")
        for nuc in range(fragment_len):
            if fragment[nuc] in ["A","C","G","T"]:
                pos = mapping[fragment[nuc]]
                fragment_onehot[nuc, pos] = 1
        one_hot_list.append(fragment_onehot)
    return np.array(one_hot_list)

def cat_encode_fragment(fragment,dna_codon_mapping):
    fragment_cat = np.zeros((3, (len(fragment) // 3), 21))

    for i in range(len(fragment) - 2):

        # 'N' means the base could not be identified due to DNA sequence quality, ignore
        # Maybe should remove the N from the sequence before encoding (scrubbing)
        if fragment[i:i+3] not in dna_codon_mapping.keys():
            continue
        codon = dna_codon_mapping[fragment[i:i+3]]

        # i % 3 selects which of the 3 start position fragments to set
        # i // 3 slides forward in each fragment after setting for each start position
        # codon indexes the ones-hot
        fragment_cat[i % 3][i // 3][codon] = 1

    return fragment_cat

def cat_encode(fragment_list):
    """
    Summary:
        Converts a list of genetic sequences into its CAT-encoding
    Parameters:
        fragment_list: list of genetic sequences
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

    encoded_sequence_array = np.zeros((len(fragment_list), 3, (len(fragment_list[0]) // 3), 21), dtype="uint8")

    for i, fragment in enumerate(fragment_list):
        encoded_sequence_array[i] = cat_encode_fragment(fragment,dna_codon_mapping)

    return encoded_sequence_array
