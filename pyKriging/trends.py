import numpy as np


def trend(type, X):
    if type is None or trend == 'constant':
        
        n = len(X)
        F = np.diag(np.ones((n,)))
        
    elif type == 'First':
        # 1, x1, x2
        F = np.array([[1] * n, [X[0]] * n, [X[1]] * n]).T
        print(F)
    elif type == 'Second':
        raise NotImplementedError
        
    elif type == 'Third':
        raise NotImplementedError
        
    elif type == 'Bspline':
        raise NotImplementedError
        
    else:
        print('Unknown trend function type')
        raise ValueError
    
    return F, F[1, :]
