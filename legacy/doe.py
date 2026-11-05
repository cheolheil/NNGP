import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, cdist, squareform


def lhd(n, m, criterion='random', X_init=None, T=10000, seed=None):
    if criterion not in ['random', 'maximin']:
        raise Exception('criterion must be random or maximin.')
        
    if seed != None:
        np.random.seed(seed)
        
    if X_init == None:
        l = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1)
        L = np.zeros((n, m))
        for i in range(m):
            L[:, i] = np.random.choice(l, n, replace=False)
        U = np.random.rand(n, m)
        X_old = (L + (n - 1) / 2 + U) / n
    else:
        X_old = X_init
        
    if criterion == 'random':
        return X_old
    elif criterion == 'maximin':
        X_new = X_old.copy()
        d_vec = pdist(X_old)
        d_mat = squareform(d_vec)
        md = d_vec[np.nonzero(d_vec)].min()

        for i in range(T):
            rows = np.argwhere(d_mat == md)[0]
            row = np.random.choice(rows, 1)
            col = np.random.choice(m)
            new_row = np.random.choice(np.delete(np.arange(n), row))
            rows = [row[0], new_row]
            X_new[rows, col] = X_new[rows[::-1], col]
            new_d = cdist(X_new[rows], X_new)
            mdprime = new_d[np.nonzero(new_d)].min()
            if mdprime > md:
                d_mat[rows, :] = new_d
                d_mat.T[rows, :] = new_d
                d_vec = squareform(d_mat)
                md = d_vec[np.nonzero(d_vec)].min()
            else:
                X_new[rows, col] = X_new[rows[::-1], col]
        return X_new


def rect_lhd(xref, n):
    lb = xref.min(axis=0)
    ub = xref.max(axis=0)
    x = lhd(n, xref.shape[1])
    x = (ub - lb) * x + lb
    return x


def factor_two_approx(X, n, method='convexhull'):
    print("Initializing...", end=" ")
    if method == 'convexhull':
        hull = ConvexHull(X)
        hull_points = X[hull.vertices, :]
        hdist = cdist(hull_points, hull_points)
        best_pair = np.unravel_index(hdist.argmax(), hdist.shape)
        print("Done!")
        S = np.array([hull_points[best_pair[0]], hull_points[best_pair[1]]])
        ids = np.array([hull.vertices[best_pair[0]], hull.vertices[best_pair[1]]])
    elif method == 'overall':
        dist_mat = cdist(X, X)
        best_pair = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        print("Done!")
        S = np.array([X[best_pair[0]], X[best_pair[1]]])
        ids = np.array([best_pair[0], best_pair[1]])
    elif method == 'sequential':
        N = len(X)
        d_max = 0
        for i in range(N-1):
            dist_arr = cdist(X[[i]], X[i+1:]).ravel()
            di_max_id = np.argmax(dist_arr)
            di_max = dist_arr[di_max_id]
            if di_max > d_max:
                d_max = di_max
                best_pair = (i, i + di_max_id + 1)
        print("Done!")
        S = np.array([X[best_pair[0]], X[best_pair[1]]])
        ids = np.array([best_pair[0], best_pair[1]])
    else:
        raise Exception('method must be convexhull or overall.')

    print("Finding optimal points...", end=' ')
    while len(S) < n:
        X_S_dist = cdist(X, S)
        new_point_idx = np.argmax(np.min(X_S_dist, axis=1))
        new_point = np.expand_dims(X[new_point_idx, :], axis=0)
        S = np.vstack((S, new_point))
        ids = np.hstack((ids, new_point_idx))
    print('Done!')
    return S, ids