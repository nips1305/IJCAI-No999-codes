import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import os
import logging
import sys

from pathlib import Path


def load_cora():
    path = ('data/cora/')
    data_name = 'cora'
    print('Loading from cora raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features.toarray(), idx_map, adj.toarray(), labels


def load_wiki(dataset):
    data = scio.loadmat('./data/wiki.mat')
    features = data['fea'].astype(float)
    adj = data['W']
    adj = adj.astype(float)
    if not sp.issparse(adj):
        adj = sp.csc_matrix(adj)
    if sp.issparse(features):
        features = features.toarray()
    labels = data['gnd'].reshape(-1) - 1
    n_classes = len(np.unique(labels))
    return adj, features, labels, n_classes

def load_pubmed():
    print('Loading from pub raw data file...')
    data = scio.loadmat('data/pubmed.mat')
    adj = data['W']
    features = data['fea']
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    return features, None, adj.tocoo(), labels


def load_citeseer():
    path = 'data/citeseer/'
    data_name = 'citeseer'
    print('Loading from citeseer raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.str)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    rows_to_delete = []
    for i in range(edges_unordered.shape[0]):
        if edges[i, 0] is None or edges[i, 1] is None:
            rows_to_delete.append(i)
    edges = np.array(np.delete(edges, rows_to_delete, axis=0), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features.toarray(), idx_map, adj.toarray(), labels

def load_graph_data(root_path=".", dataset_name="dblp", show_details=False):
    """
    load graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :returns feat, label, adj: the features, labels and adj
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        stream=sys.stdout)
    root_path = root_path + "data/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset_path = root_path + dataset_name
    if not os.path.exists(dataset_path):
        # down load
        url = "https://drive.google.com/file/d/1_LesghFTQ02vKOBUfDP8fmDF1JP3MPrJ/view?usp=sharing"
        logging.info("Downloading " + dataset_name + " dataset from: " + url)
    else:
        logging.info("Loading " + dataset_name + " dataset from local")
    load_path = root_path + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)

    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("edge num:   ", int(adj.sum() / 2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj

def load_data(name):
    if name.lower() == 'cora':
        features, _, adj, labels = load_cora()
        return features, adj, labels
    # elif name.lower() == 'citeseer':
    #     features, _, adj, labels = load_citeseer()
    #     return features, adj, labels
    # elif name.lower() == 'wiki':
    #     features, _, adj, labels = load_wiki()
    #     return features, adj, labels

    # --------npy---------------
    elif name.lower() == 'dblp':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="dblp", show_details=True) 
        return features, adj, labels

    elif name.lower() == 'acm':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="acm", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'citeseer':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="citeseer", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'bat':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="bat", show_details=True) 
        return features, adj, labels
    elif name.lower() == 'uat':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="uat", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'cornell':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="cornell", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'texas':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="texas", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'eat':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="eat", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'wisc':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="wisc", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'film':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="film", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'amap':
        features, labels, adj = load_graph_data(root_path='./', dataset_name="amap", show_details=True)  
        return features, adj, labels
    elif name.lower() == 'pubmed':
        features, _, adj, labels = load_pubmed()
        return features, adj, labels
    elif name.lower() == 'mgae_citeseer':
        path = 'data/{}.mat'.format(name)
        data = scio.loadmat(path)
        labels = data['gnd']
        labels = np.reshape(labels, (labels.shape[0],))
        adj = data['W']
        return data['fea'], adj.toarray(), labels
    elif name.lower() == 'wiki':
        print('wiki')
        adj, features, labels, n_classes = load_wiki('wiki')  # dblp
        return features, adj.toarray(), labels



