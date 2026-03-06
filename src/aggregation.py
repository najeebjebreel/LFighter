import copy
import enum
import torch
import numpy as np
import math
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
import hdbscan
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from utils import *



def get_pca(data, threshold = 0.99):
    normalized_data = StandardScaler().fit_transform(data)
    pca = PCA()
    reduced_data = pca.fit_transform(normalized_data)
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)
    select_pcas = np.where(cum_sum_eigenvalues <=threshold)[0]
    # print('Number of components with variance <= {:0.0f}%: {}'.format(threshold*100, len(select_pcas)))
    reduced_data = reduced_data[:, select_pcas]
    return reduced_data

eps = np.finfo(float).eps

class LFD():
    def __init__(self, num_classes):
        self.memory = np.zeros([num_classes])
    
    def clusters_dissimilarity(self, clusters):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def aggregate(self, global_model, local_models, ptypes):
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i]= global_model[-2].cpu().data.numpy() - \
                local_models[i][-2].cpu().data.numpy() 
            db[i]= global_model[-1].cpu().data.numpy() - \
                local_models[i][-1].cpu().data.numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        "If one class or two classes classification model"
        if len(db[0]) <= 2:
            data = []
            for i in range(m):
                data.append(dw[i].reshape(-1))
        
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            labels = kmeans.labels_

            clusters = {0:[], 1:[]}
            for i, l in enumerate(labels):
                clusters[l].append(data[i])

            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:
                good_cl = 1

            # print('Cluster 0 weighted variance', cs0)
            # print('Cluster 1 weighted variance', cs1)
            # print('Potential good cluster is:', good_cl)
            scores = np.ones([m])
            for i, l in enumerate(labels):
                # print(ptypes[i], 'Cluster:', l)
                if l != good_cl:
                    scores[i] = 0
                
            global_weights = average_weights(local_weights, scores)
            return global_weights

        "For multiclassification models"
        norms = np.linalg.norm(dw, axis = -1) 
        self.memory = np.sum(norms, axis = 0)
        self.memory +=np.sum(abs(db), axis = 0)
        max_two_freq_classes = self.memory.argsort()[-2:]
        print('Potential source and target classes:', max_two_freq_classes)
        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1:[]}
        for i, l in enumerate(labels):
          clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        # print('Cluster 0 weighted variance', cs0)
        # print('Cluster 1 weighted variance', cs1)
        # print('Potential good cluster is:', good_cl)
        scores = np.ones([m])
        for i, l in enumerate(labels):
            # print(ptypes[i], 'Cluster:', l)
            if l != good_cl:
                scores[i] = 0
            
        global_weights = average_weights(local_weights, scores)
        return global_weights

################################################
# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
       
    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers]+= grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]


#######################################################################################
class Tolpegin:
    def __init__(self):
        pass
    
    def score(self, global_model, local_models, peers_types, selected_peers):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grad= (last_g - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy())
            grads[i] = grad
        
        grads = np.array(grads)
        num_classes = grad.shape[0]
        # print('Number of classes:', num_classes)
        dist = [ ]
        labels = [ ]
        for c in range(num_classes):
            data = grads[:, c]
            data = get_pca(copy.deepcopy(data))
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            cl = kmeans.cluster_centers_
            dist.append(((cl[0] - cl[1])**2).sum())
            labels.append(kmeans.labels_)
        
        dist = np.array(dist)
        candidate_class = dist.argmax()
        print("Candidate source/target class", candidate_class)
        labels = labels[candidate_class]
        if sum(labels) < m/2:
            scores = 1 - labels
        else:
            scores = labels
        
        for i, pt in enumerate(peers_types):
            print(pt, 'scored', scores[i])
        return scores
#################################################################################################################
# Clip local updates
def clipp_model(g_w, w, gamma =  1):
    for layer in w.keys():
        w[layer] = g_w[layer] + (w[layer] - g_w[layer])*min(1, gamma)
    return w
def FLAME(global_model, local_models, noise_scalar):
    # Compute number of local models
    m = len(local_models)
    
    # Flattent local models
    g_m = np.array([torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()])
    f_m = np.array([torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models])
    grads = g_m - f_m
    # Compute model-wise cosine similarity
    cs = smp.cosine_similarity(grads)
    # Compute the minimum cluster size value
    msc = int(m*0.5) + 1 
    # Apply HDBSCAN on the computed cosine similarities
    clusterer = hdbscan.HDBSCAN(min_cluster_size=msc, min_samples=1, allow_single_cluster = True)
    clusterer.fit(cs)
    labels = clusterer.labels_
    # print('Clusters:', labels)

    if sum(labels) == -(m):
        # In case all of the local models identified as outliers, consider all of as benign
        benign_idxs = np.arange(m)
    else:
        benign_idxs = np.where(labels!=-1)[0]
        
    # Compute euclidean distances to the current global model
    euc_d = cdist(g_m, f_m)[0]
    # Identify the median of computed distances
    st = np.median(euc_d)
    # Clipp admitted updates
    W_c = []
    for i, idx in enumerate(benign_idxs):
        w_c = clipp_model(global_model.state_dict(), local_models[idx].state_dict(), gamma =  st/euc_d[idx])
        W_c.append(w_c)
    
    # Average admitted clipped updates to obtain a new global model
    g_w = average_weights(W_c, np.ones(len(W_c)))
    
    '''From the original paper: {We use standard DP parameters and set eps = 3705 for IC, 
    eps = 395 for the NIDS and eps = 4191 for the NLP scenario. 
    Accordingly, lambda = 0.001 for IC and NLP, and lambda = 0.01 for the NIDS scenario.}
    However, we found lambda = 0.001 with the CIFAR10-ResNet18 benchmark spoils the model
    and therefore we tried lower lambda values, which correspond to greater eps values.'''
    
    # Add adaptive noise to the global model
    lamb = 0.001
    sigma = lamb*st*noise_scalar
    # print('Sigma:{:0.4f}'.format(sigma))
    for key in g_w.keys():
        noise = torch.FloatTensor(g_w[key].shape).normal_(mean=0, std=(sigma**2)).to(g_w[key].device)
        g_w[key] = g_w[key] + noise
        
    return g_w 
#################################################################################################################

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med
        
# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])
        
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg
   
def Krum(updates, f, multi = False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]
##################################################################
