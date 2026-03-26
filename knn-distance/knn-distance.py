import numpy as np
def dist(a,b):
    return np.sqrt(np.sum(np.square(a - b)))
def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    ans = np.ndarray(shape = (X_test.shape[0],k),dtype = np.int32)
    for i in range(X_test.shape[0]):
        res = np.ndarray(shape = (X_train.shape[0],),dtype = np.float64)
        id = 0
        for j in range(X_train.shape[0]):
            res[id] = dist(X_test[i],X_train[j])
            id += 1
        res = np.argsort(res)
        if res.shape[0] < k:
            tmp = np.full(shape = (k - res.shape[0],),fill_value = -1)
            res = np.concatenate((res,tmp))
        ans[i] = res[:k]
    return np.array(ans)