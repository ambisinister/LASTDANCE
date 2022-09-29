import pickle
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

trajectories = pickle.load(open('./trajectories.pkl', 'rb'))
penult = pickle.load(open('./penult_features.pkl', 'rb'))

tr_X, y = trajectories
tr_X = np.array(tr_X)
pe_X, _ = penult
pe_X = np.array(pe_X)

# Flatten trajectories, since shape is (n, L, 2)
tr_X = np.reshape(tr_X, (tr_X.shape[0], np.prod(tr_X.shape[1:])))


is_anomaly = [1 if x == 10 else 0 for x in y]


def get_gaussians(feats, labels, n_classes=10):
    # Fit Class Conditional Gaussians
    # Do not fit a gaussian over ood data
    
    gaussians = []
    for i in range(n_classes):
        x_ci = [x for x,y in zip(feats, labels) if y == i]
        mu = np.mean(x_ci, axis=0)
        cov = np.cov(x_ci, rowvar=0)
        inv = np.linalg.pinv(cov)
        gaussian = {
            'class': i,
            'mean': mu,
            'covariance': cov,
            'inv_covariance': inv
        }
        gaussians.append(gaussian)

    return gaussians

def scaled_mahalanobis(feats, gaussians):
    dists = []

    for f in feats:
        min_dist = math.inf
        for g in gaussians:
            d = distance.mahalanobis(f, g['mean'], g['inv_covariance'])
            if d < min_dist:
                min_dist = d
        dists.append(min_dist)

    return MinMaxScaler().fit_transform(np.reshape(dists, (-1, 1)))
        
gaussians_pen = get_gaussians(pe_X, y)
gaussians_tra = get_gaussians(tr_X, y)
tra_dists = scaled_mahalanobis(tr_X, gaussians_tra)
pen_dists = scaled_mahalanobis(pe_X, gaussians_pen)

print(f"Mean dist for trajectory: IID {np.mean(tra_dists[:10000])}, OOD {np.mean(tra_dists[10000:])}")
print(f"Mean dist for penultimate: IID {np.mean(pen_dists[:10000])}, OOD {np.mean(pen_dists[10000:])}")
