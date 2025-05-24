import numpy as np

# 20 種標準胺基酸
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

def aac_feature(sequence):
    seq = ''.join([aa for aa in sequence.upper() if aa in AMINO_ACIDS])
    if len(seq) == 0:
        return np.zeros(20)  # 防呆：全部是奇怪的字母
    aac = [seq.count(aa) / len(seq) for aa in AMINO_ACIDS]
    return np.array(aac)
