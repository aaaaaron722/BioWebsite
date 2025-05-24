import numpy as np
import os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import shutil

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

def get_aa_index(aa):
    return AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else None

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

def pssm_feature(sequence: str, tmp_dir='tmp', db_path='/Users/linyouxun/Documents/其他作業/生物資訊概論/BioWeb/BioWebsite/data/db/swissprot'):
    os.makedirs(tmp_dir, exist_ok=True)

    # 存 fasta
    fasta_path = os.path.join(tmp_dir, 'query.fasta')
    with open(fasta_path, 'w') as f:
        record = SeqRecord(Seq(sequence), id='query', description='')
        SeqIO.write(record, f, 'fasta')

    # 跑 psiblast
    cmd = [
        'psiblast',
        '-query', fasta_path,
        '-db', db_path,
        '-num_iterations', '3',
        '-out_ascii_pssm', os.path.join(tmp_dir, 'query.pssm'),
        '-num_threads', '8'
    ]
    subprocess.run(cmd, check=True)

    # 讀取 PSSM
    pssm_file = os.path.join(tmp_dir, 'query.pssm')
    with open(pssm_file) as f:
        lines = f.readlines()[3:-6]  # 去掉頭尾的廢話

    matrix = np.zeros((20, 20))  # 初始化 20x20 矩陣

    for i, line in enumerate(lines):
        if i >= len(sequence):
            break
        values = list(map(float, line.split()[2:22]))
        aa = sequence[i]
        aa_idx = get_aa_index(aa)
        if aa_idx is not None:
            matrix[aa_idx] += values  # 同種胺基酸累加

    matrix = sigmoid(matrix)  # 把分數轉到 0~1 區間

    shutil.rmtree(tmp_dir)  # 清乾淨 tmp 檔案

    return matrix  # shape = (20, 20)
