import torch

def compute_empirical_means(data, labels):
    means = []
    labels_unique=torch.unique(labels)
    for label in labels_unique:
        data_label=data[labels==label]
        means.append(torch.mean(data_label, axis=0))
    return torch.stack(means) # n x d

def pairwise_dist(X, P):
    X2 = (X ** 2).sum(dim=1, keepdim=True) # (n x 1)                                                            
    P2 = (P ** 2).sum(dim=1, keepdim=True) # (n' x 1)                                                            
    XP = X @ P.T # (n x n')                                                                                      
    return X2 + P2.T - 2 * XP # (n x n')  
    
def compute_empirical_cov_matrix(data, labels, means):
    N = len(data)
    empirical_cov = 0
    labels_unique=torch.unique(labels)
    for i in range(labels_unique.shape[0]):
        mean_label = means[i, :].reshape((1, -1)) # [1, d]
        label = i
        data_label = data[labels==label]
        dist_sq = pairwise_dist(data_label, mean_label) 
        empirical_cov += torch.sum(dist_sq)
    return empirical_cov/N

def mahalanobis_test(data, means, cov):
    dist_sq  = torch.subtract(data[:, None, :], means[None, :, :])**2 # [n, n', d]
    dist_sq = -1*torch.sum(dist_sq, dim=2)/cov # [n, n']
    return torch.max(dist_sq, dim=1)[0]