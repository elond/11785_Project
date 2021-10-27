from Bio import SeqIO

def readin_fasta(input_file):
    """
    Summary:
        Reads a fasta file into two lists. One contains all sequence names while
        the other contains all genetic sequences
    Parameters:
        input_file: Path to input file
    Returns:
        fasta_name_list: List of sequence names
        sequence_list: List of genetic sequences
    """
    fasta_name_list = []
    sequence_list = []

    for seq_record in SeqIO.parse(input_file, "fasta"):
        fasta_name_list.append(seq_record.id)
        sequence_list.append(str(seq_record.seq))

    return fasta_name_list, sequence_list
