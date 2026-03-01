import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    padded = []
    for seq in seqs:
        if len(seq) == max_len:
            padded.append(seq)
            continue
        else:
            diff = max_len  - len(seq)
            if diff > 0:
                for i in range(diff):
                    seq.append(pad_value)
                padded.append(seq)
            elif diff <0:
                diff = -1*diff
                for i in range(diff):
                    seq.pop(-1)
                padded.append(seq)
                    
    return padded
    
