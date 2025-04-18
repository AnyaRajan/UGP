from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
import os
import pandas as pd
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, Subset,ConcatDataset
from sklearn.model_selection import train_test_split
import models
import numpy as np
# import corrupted_text
from torch.utils.data import TensorDataset, DataLoader
import random
import re
from sklearn.preprocessing import LabelEncoder
from omegaconf import OmegaConf
from pathlib import Path
conf = OmegaConf.load('config.yaml')
from nltk.corpus import stopwords
import dataclasses
from torchvision import transforms
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
from PIL import Image
# # # # # # # # #
# def get_augmentation_pipeline():
#     return transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010))
#     ])

def get_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.2)),  # Randomly scale images
        transforms.RandomHorizontalFlip(p=0.5),  # Flip with 50% probability
        transforms.RandomVerticalFlip(p=0.2),  # Vertical flip with 20% probability
        transforms.RandomRotation(30),  # Rotate up to ±30 degrees
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Adjust color
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # Perspective shifts
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Introduce warping
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Random blurring
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# def get_augmentation_pipeline():
#     return transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),  # for CIFAR-10
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor()
#     ])

    
def ensemble_scores(*score_arrays):
    return np.mean(score_arrays, axis=0)

def compute_class_weights(y_train_tensor, device):
    y_train_np = y_train_tensor.cpu().numpy()
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate_models(model_scores_dict, ground_truth_flags):
    summary = []
    bug_count = int(np.sum(ground_truth_flags))
    for name, scores in model_scores_dict.items():
        ranking = np.argsort(scores)[::-1]
        sorted_flags = ground_truth_flags[ranking]
        rauc_100 = rauc(sorted_flags, 100)
        rauc_200 = rauc(sorted_flags, 200)
        rauc_500 = rauc(sorted_flags, 500)
        rauc_1000 = rauc(sorted_flags, 1000)
        rauc_all = rauc(sorted_flags, len(sorted_flags))
        atrc_score, _ = ATRC(sorted_flags, bug_count)
        score = 0.5 * rauc_100 + 0.3 * rauc_500 + 0.2 * atrc_score  # Composite weighted score
        summary.append({
            "name": name,
            "RAUC@100": rauc_100,
            "RAUC@200": rauc_200,
            "RAUC@500": rauc_500,
            "RAUC@1000": rauc_1000,
            "RAUC@all": rauc_all,
            "ATRC": atrc_score,
            "Composite Score": score,
            "Sorted Flags": sorted_flags
        })
    return summary

@dataclasses.dataclass
class CorruptionWeights:
    """Configuration of the weights of the different corruption types."""

    typo_weight: float = 0
    autocomplete_weight: float = 0
    autocorrect_weight: float = 0
    synonym_weight: float = 0

    def set_weights(self, corruption):
        if corruption=="typo":
            self.typo_weight=1
        elif corruption=="autocomplete":
            self.autocomplete_weight=1
        elif corruption=="autocorrect":
            self.autocorrect_weight=1
        elif corruption=="synonym":
            self.synonym_weight=1
        else:
            raise ValueError("Invalid corruption type")
    def __hash__(self):
        return hash(tuple(dataclasses.asdict(self).values()))

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train, y_train, x_val, y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]
    return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(
        encoded_test), onehot_dict

def tockenize_SMS(x_train, x_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    return np.array(final_list_train), np.array(final_list_test), onehot_dict

class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        label = self.label[index]
        return sample, label

def get_num_of_most_diff_class(labels):
    classes=labels.transpose()
    target_class=classes[:,-1]
    max_different_count=np.zeros(classes.shape[0],dtype=int)
    for row_idx,row in enumerate(classes):
        different_values,counts=np.unique(row[:-1][row[:-1]!=target_class[row_idx]],return_counts=True)

        if counts.size>0:
            max_count_index=np.argmax(counts)
            max_different_count[row_idx]=counts[max_count_index]
        else:
            max_different_count[row_idx]=0

    return max_different_count

def calculate_label_std(labels):
    std = np.std(labels[:,conf.start:], axis=1)
    return std

def calculate_avg_info(infos):
    infos=infos.transpose()
    avg = np.mean(infos[:,conf.start:], axis=1)
    return avg

def calculate_avg_pro_diff(pros):
    p=pros.transpose((1,0,2))
    target=p[:,-1,:]
    average_distances=np.zeros(p.shape[0],dtype=float)
    for id,row in enumerate(p):
        cosine_similarities = cosine_similarity(row[:-1, :], target[id].reshape(1, -1))
        average_distances[id] = np.mean(1 - cosine_similarities)
    return average_distances

def calculate_std_info(infos):
    infos=infos.transpose()
    std = np.std(infos[:,conf.start:], axis=1)
    return std

def ATRC(sorted,budget=conf.budget):
    total=0
    trcs=[]
    for i in range(1,budget):
        trc=TRC(sorted,i)
        total+=trc
        trcs.append(trc)

    for i in range(budget,len(sorted)):
        trc=TRC(sorted,i)
        trcs.append(trc)
    return total/(budget-1),trcs

def TRC(sorted, budget=conf.budget):
    num_bug=len(np.where(sorted == 1)[0])
    tmp=sorted[:budget]
    bug_index= np.where(tmp == 1)[0]
    m=len(bug_index)
    TRC=m/min(budget,num_bug)
    return TRC


def rauc(sorted, num=conf.num):
    sorted=sorted[:num]
    bug_index = np.where(sorted == 1)[0]
    n = len(sorted)
    m = len(bug_index)
    if m==0:
        return 0
    n_0=0
    true=[]
    for i in sorted:
        if i==1:
            n_0=n_0+1
            true.append(n_0)
        else:
            true.append(n_0)
    ideal=(n-m)*m+(m+1)*m/2
    RAUC=np.sum(true)/ideal
    return RAUC

def get_retrain_data(indicies,dataset=conf.dataset,corruption=conf.corruption):
    trainloader=get_train_data(dataset)
    train_dataset=trainloader.dataset
    test_dataset=get_corruption_test_data(corruption).dataset
    selected_test_data=Subset(test_dataset,indicies)
    new_train_data=ConcatDataset([train_dataset,selected_test_data])
    retrain_loader=DataLoader(new_train_data,shuffle=True,batch_size=trainloader.batch_size,num_workers=4)
    return retrain_loader

def get_clean_test_dataset(dataset):
    testloader = None
    if dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root=conf.data_root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    # elif dataset == 'imagenet':
    #     transform_train = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((64, 64)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406),
    #                              (0.229, 0.224, 0.225)),
    #     ])
    #     test_data = np.load(conf.data_root + '/tiny-imagenet-200/test_data.npy')
    #     test_label = np.load(conf.data_root + '/tiny-imagenet-200/test_label.npy')
    #     trainset = CustomDataset(test_data, test_label, transform=transform_train)
    #     testloader = torch.utils.data.DataLoader(
    #         trainset, batch_size=60, shuffle=False, num_workers=4, pin_memory=True)
    # elif dataset == 'imdb':
    #     x_train = np.load("dataset/IMDB/train_data.npy", allow_pickle=True)
    #     y_train = np.load("dataset/IMDB/train_label.npy", allow_pickle=True)
    #     x_test = np.load("dataset/IMDB/test_data.npy", allow_pickle=True)
    #     y_test = np.load("dataset/IMDB/test_label.npy", allow_pickle=True)
    #     x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)


    #     x_train_pad = padding_(x_train, 500)
    #     x_test_pad = padding_(x_test, 500)

    #     train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    #     valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    #     batch_size = 50

    #     train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=4, batch_size=batch_size)
    #     testloader = torch.utils.data.DataLoader(valid_data, shuffle=False, num_workers=4, batch_size=batch_size)
    # elif dataset =="SMS":
    #     data_root = Path("dataset/SMS")

    #     x_train = np.load(data_root / Path("train_data.npy"), allow_pickle=True)
    #     y_train = np.load(data_root / Path("train_label.npy"), allow_pickle=True)
    #     x_test = np.load(data_root / Path("test_data.npy"), allow_pickle=True)
    #     y_test = np.load(data_root / Path("test_label.npy"), allow_pickle=True)
    #     x_train, x_test, vocab = tockenize_SMS(x_train, x_test)

    #     # x_train_pad = padding_(x_train, 150)
    #     x_test_pad = padding_(x_test, 55)
    #     # train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    #     valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    #     # train_batch_size = 4
    #     test_batch_size = 50

    #     # train_loader = DataLoader(train_data, shuffle=True, num_workers=4, batch_size=train_batch_size)
    #     testloader = torch.utils.data.DataLoader(valid_data, shuffle=False, num_workers=4, batch_size=test_batch_size)
    
    return testloader

def get_corruption_test_data(corruption=conf.corruption):
    testloader = None
    if conf.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_data = np.load("dataset/cifar-10-c/" + corruption + ".npy")[
                    10000 * (conf.severity - 1):10000 * conf.severity]
        test_label = np.load("dataset/cifar-10-c/labels.npy")[10000 * (conf.severity - 1):10000 * conf.severity]
        testset = CustomDataset(test_data, test_label, transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    # elif conf.dataset == 'imagenet':
    #     transform_train = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((64, 64)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406),
    #                              (0.229, 0.224, 0.225)),
    #     ])
    #     test_data = np.load(conf.data_root + '/tiny-imagenet-200/' + corruption + '_test_data.npy')
    #     test_label = np.load(conf.data_root + '/tiny-imagenet-200/' + corruption + '_test_label.npy')
    #     trainset = CustomDataset(test_data, test_label, transform=transform_train)
    #     testloader = torch.utils.data.DataLoader(
    #         trainset, batch_size=30, shuffle=False, num_workers=4, pin_memory=True)
    # elif conf.dataset == 'imdb':
    #     data_root=Path("dataset/IMDB")
    #     x_train = np.load(data_root / Path("train_data.npy"), allow_pickle=True)
    #     y_train = np.load(data_root / Path("train_label.npy"), allow_pickle=True)
    #     x_test = np.load(data_root / Path("test_data.npy"), allow_pickle=True)
    #     y_test = np.load(data_root / Path("test_label.npy"), allow_pickle=True)
    #     if (data_root / Path(corruption + "_test_data.npy")).exists():
    #         x_test = np.load(data_root / Path(corruption + "_test_data.npy"), allow_pickle=True)
    #     else:
    #         x_train, x_test = x_train.tolist(), x_test.tolist()
    #         corruptor = corrupted_text.TextCorruptor(base_dataset=x_test + x_train,
    #                                                  cache_dir=".mycache")
    #         weights = CorruptionWeights()
    #         weights.set_weights(corruption)
    #         imdb_corrupted = corruptor.corrupt(x_test, severity=1, seed=42, weights=weights)
    #         x_train = np.array(x_train)
    #         x_test = np.array(imdb_corrupted)
    #         np.save(data_root / Path(corruption + "_test_data.npy"), x_test)

    #     if (data_root / Path(corruption + "_test_data_pad.npy")).exists():
    #         x_test_pad = np.load(data_root / Path(corruption + "_test_data_pad.npy"), allow_pickle=True)
    #         y_test_pad=np.load(data_root / Path(corruption + "_test_label_pad.npy"), allow_pickle=True)
    #     else:
    #         x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)
    #         y_test_pad=y_test
    #         x_test_pad = padding_(x_test, 500).astype(np.int32)
    #         np.save(data_root / Path(corruption + "_test_data_pad.npy"), x_test_pad)
    #         np.save(data_root / Path(corruption + "_test_label_pad.npy"), y_test_pad)
    #     # train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    #     test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test_pad))
    #     testloader=DataLoader(test_data, shuffle=False, num_workers=4, batch_size=50)
    # elif conf.dataset =='SMS':
    #     data_root = Path("dataset/SMS")

    #     x_train = np.load(data_root / Path("train_data.npy"), allow_pickle=True)
    #     y_train = np.load(data_root / Path("train_label.npy"), allow_pickle=True)
    #     x_test = np.load(data_root / Path("test_data.npy"), allow_pickle=True)
    #     y_test = np.load(data_root / Path("test_label.npy"), allow_pickle=True)
    #     if (data_root/Path(corruption+"_test_data.npy")).exists():
    #         x_test = np.load(data_root / Path(corruption+"_test_data.npy"), allow_pickle=True)
    #     else:
    #         x_train, x_test = x_train.tolist(), x_test.tolist()
    #         corruptor = corrupted_text.TextCorruptor(base_dataset=x_test + x_train,
    #                                                      cache_dir=".mycache")
    #         weights = CorruptionWeights()
    #         weights.set_weights(corruption)
    #         imdb_corrupted = corruptor.corrupt(x_test, severity=1, seed=42, weights=weights)
    #         x_train = np.array(x_train)
    #         x_test = np.array(imdb_corrupted)
    #         np.save(data_root/Path(corruption+"_test_data.npy"),x_test)
    #     x_train, x_test, vocab = tockenize_SMS(x_train, x_test)

    #     x_test_pad = padding_(x_test, 55)

    #     valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))
    #     test_batch_size = 50

    #     testloader = DataLoader(valid_data, shuffle=False, num_workers=4, batch_size=test_batch_size)
    
    return testloader

def get_train_data(dataset):
    trainloader = None
    if dataset == 'cifar10':
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                          (0.2023, 0.1994, 0.2010)),
        # ])
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=conf.data_root, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    else:
        raise ValueError('Dataset not found')
    return trainloader

def get_val_and_test(corruption=conf.corruption,ratio=0.75):
    testloader = None
    valloader = None
    if conf.dataset == 'cifar10':
        # Use standard test transforms for CIFAR10
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        # Load the clean CIFAR10 test set using torchvision
        import torchvision.datasets as datasets
        ori_testset = datasets.CIFAR10(root=conf.data_root, train=False, download=True, transform=transform_test)
        indices = np.arange(len(ori_testset))
        np.random.shuffle(indices)
        split = int(np.floor(ratio * len(ori_testset)))
        test_indices, val_indices = indices[:split], indices[split:]
        from torch.utils.data import Subset
        testset = Subset(ori_testset, test_indices)
        valset = Subset(ori_testset, val_indices)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    
    return valloader, testloader
