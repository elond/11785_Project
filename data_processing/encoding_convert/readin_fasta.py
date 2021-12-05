from Bio import SeqIO
import sys

def readin_fasta(input_file, batch_size):
    """Read fasta file with a fast, memory-efficient generator."""
    title_list = []
    seq_list = []
    seq_num = len([1 for line in open(input_file) if line.startswith(">")])

    for i, seq_record in enumerate(SeqIO.FastaIO.SimpleFastaParser(open(input_file)),1):
        title, seq = seq_record
        title_list.append(title)
        seq_list.append(seq)
        if i%batch_size == 0:
            yield title_list, seq_list
            title_list = []
            seq_list = []
        if i == seq_num:
            print('Converted {} of {} fragments'.format(i, seq_num))
            yield title_list, seq_list
