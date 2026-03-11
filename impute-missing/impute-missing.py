import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """

    X = np.array(X)
    was_1d = False
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        was_1d = True
    all_column_stat = []
    for i in range(X.shape[-1]):
        all_nums = [float(x) for x in X[:,i] if not np.isnan(x)]
        if len(all_nums) == 0:
            all_column_stat.append(0.0)
        elif strategy == 'mean': 
            all_column_stat.append(np.mean(all_nums))
        else:
            all_column_stat.append(np.median(all_nums))
    for i in range(X.shape[-1]):
        new_tensor = []
        for val in X[:,i]:
            if np.isnan(val):
                new_tensor.append(all_column_stat[i])
            else:
                new_tensor.append(val)
        X[:,i] = np.asarray(new_tensor, dtype='float')

    if was_1d:
        return X.ravel()
    return np.asarray(X, dtype = 'float')