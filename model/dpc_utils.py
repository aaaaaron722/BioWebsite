import numpy as np
def dpc_feature(sequence):
    # 20 種胺基酸
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq = sequence.upper()
    dpc_dict = {aa1+aa2:0 for aa1 in amino_acids for aa2 in amino_acids}
    total = len(seq) - 1
    if total <= 0:
        return np.zeros(400)
    for i in range(total):
        dipeptide = seq[i:i+2]
        if dipeptide in dpc_dict:
            dpc_dict[dipeptide] += 1
    dpc = [dpc_dict[aa1+aa2]/total for aa1 in amino_acids for aa2 in amino_acids]
    return np.array(dpc)