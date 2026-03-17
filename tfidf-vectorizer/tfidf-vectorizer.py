import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    all_doc_freqs = []
    all_words = []
    for doc in documents:
        counted = Counter(doc.split())
        all_doc_freqs.append(counted)
        all_words.extend(doc.split())

    vocab= sorted(list(set(all_words)))

    idf = {}
    for w in vocab:
        count = 0
        for doc in all_doc_freqs:
            if doc.get(w, None) is None:
                continue
            else:
                count+=1
        idf[w] = math.log(len(documents)/count)

    
    tfidf_matrix = np.zeros((len(documents), len(vocab)))
    for i in range(len(vocab)):
        for j in range(len(documents)):
            tf = (all_doc_freqs[j].get(vocab[i], 0))/sum(all_doc_freqs[j].values())
            tfidf_matrix[j,i] =  tf*idf[vocab[i]]

    return tfidf_matrix, vocab
    
    
    

    