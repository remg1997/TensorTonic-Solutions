import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    num_docs = len(docs)
    if num_docs ==0:
        return np.array([])
    total = 0
    for doc in docs:
        total += len(doc)
    avgdl = total/num_docs

    term_frequency = []
    for doc in docs:
        term_frequency.append(Counter(doc))
        
    dt = {}
    for doc in term_frequency:
        for k,v in doc.items():
            if dt.get(k, None) is None:
                dt[k] = 1
            else:
                dt[k] += 1
    #IDF
    idf_per_tok = {}
    for token in query_tokens:
        idf_per_tok[token] = math.log((num_docs - dt.get(token, 0)+0.5)/(dt.get(token, 0)+0.5)+1)

    scores = []
    for i, doc in enumerate(docs):
        per_query_scores = []
        for query in query_tokens:
            tf_q = term_frequency[i].get(query, 0)
            if tf_q ==0:
                continue
            num = tf_q*(k1+1)
            den = tf_q + k1*((1-b)+b*(len(doc)/avgdl))
            idf_q = idf_per_tok[query]
            per_query_scores.append(idf_q*(num/den))
        final_score = sum(per_query_scores)
        scores.append(final_score)
    return np.array(scores)
            
        
    
        
    
    

                    
            
        
        