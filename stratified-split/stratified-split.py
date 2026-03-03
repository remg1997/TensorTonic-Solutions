import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    unique_classes = np.unique(y)
    total = len(y)
    print(f"We have a total of {total} samples, from {len(unique_classes)} classes")
    proportions = {}
    for u_cl in unique_classes:
        counter = 0
        for s in y:
            if s == u_cl:
                counter+=1

        prop = counter/total
        proportions[int(u_cl)] = prop
    if rng is None:
        rng = np.random.default_rng()
    per_class_feat_lab = {}
    for cl in unique_classes:
        idxs = [i for i, te in enumerate(y) if te == cl]
        rng.shuffle(idxs)
        per_class_feat_lab[cl] = idxs
    test_num = np.round(test_size*total)
    prop_totals = {}
    for k,v in proportions.items():
        prop_totals[k] = np.round(np.max([1.0,v*test_num]))
      
    x_train, y_train, x_test, y_test = [],[],[],[]
    for k in unique_classes:
        idxs = per_class_feat_lab[k]
        n_test = int(np.round(test_size * len(idxs)))
    
        for i in sorted(idxs[:n_test]):
            x_test.append(X[i])
            y_test.append(int(k))
    
        for i in sorted(idxs[n_test:]):
            x_train.append(X[i])
            y_train.append(int(k))
      
    return (x_train,x_test,y_train, y_test)
    

    
        
    
            