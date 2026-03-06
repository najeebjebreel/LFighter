'''Some helper functions
'''
import random
from random import shuffle
random.seed(7)
import numpy as np
from torchvision import datasets, transforms
import codecs
import tensorflow as tf 
import pandas as pd
from datasets import *

def distribute_dataset(dataset_name, num_peers, num_classes, dd_type = 'IID', classes_per_peer = 1, samples_per_class = 582, 
alpha = 1):
    print("--> Loading of {} dataset".format(dataset_name))
    tokenizer = None
    if dataset_name == 'MNIST':
        trainset, testset = get_mnist()
    elif dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10()
    elif dataset_name == 'IMDB':
        trainset, testset, tokenizer = get_imdb(num_peers = num_peers)
    if dd_type == 'IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=1000000)
    elif dd_type == 'NON_IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        peers_data_dict = sample_extreme(trainset, num_peers, num_classes, classes_per_peer, samples_per_class)

    print("--> Dataset has been loaded!")
    return trainset, testset, peers_data_dict, tokenizer
    

# Get the original MNIST data set
def get_mnist():
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    trainset = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True,
                        transform=transform)
    return trainset, testset

# Get the original CIFAR10 data set
def get_cifar10():
    data_dir = 'data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    testset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    return trainset, testset

#Get the IMDB data set
def get_imdb(num_peers = 10):
    MAX_LEN = 128
    # Read data
    df = pd.read_csv('data/imdb.csv')
    # Convert sentiment columns to numerical values
    df.sentiment = df.sentiment.apply(lambda x: 1 if x=='positive' else 0)
    # Tokenization
    # use tf.keras for tokenization,  
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

        
    train_df = df.iloc[:40000].reset_index(drop=True)
    valid_df = df.iloc[40000:].reset_index(drop=True)

    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training
    trainset = IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    # initialize dataset class for validation
    testset = IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)
   
    return trainset, testset, tokenizer


def sample_dirichlet(dataset, num_users, alpha=1):
    classes = {}
    for idx, x in enumerate(dataset):
        _, label = x
        if type(label) == torch.Tensor:
            label = label.item()
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())
    
    peers_data_dict = {i: {'data':np.array([]), 'labels':[]} for i in range(num_users)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))
        for user in range(num_users):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            peers_data_dict[user]['data'] = np.concatenate((peers_data_dict[user]['data'], np.array(sampled_list)), axis=0)
            if num_imgs > 0:
                peers_data_dict[user]['labels'].append((n, num_imgs))
                
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]
   
    return peers_data_dict


def sample_extreme(dataset, num_users, num_classes, classes_per_peer, samples_per_class):
    n = len(dataset)
    num_classes = 10
    peers_data_dict = {i: {'data':np.array([]), 'labels':[]} for i in range(num_users)}
    idxs = np.arange(n)
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    label_indices = {l:[] for l in range(num_classes)}
    for l in label_indices:
        label_idxs = np.where(labels == l)
        label_indices[l] = list(idxs[label_idxs])
    
    labels = [i for i in range(num_classes)]

    for i in range(num_users):
        user_labels = np.random.choice(labels, classes_per_peer, replace=False)
        for l in user_labels:
            peers_data_dict[i]['labels'].append(l)
            lab_idxs = label_indices[l][:samples_per_class]
            label_indices[l] = list(set(label_indices[l])-set(lab_idxs))
            if len(label_indices[l]) < samples_per_class:
                labels = list(set(labels)-set([l]))
            peers_data_dict[i]['data'] = np.concatenate(
            (peers_data_dict[i]['data'], lab_idxs), axis=0)
    
    return peers_data_dict
        

