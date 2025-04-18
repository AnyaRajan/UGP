from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
from sklearn.preprocessing import MinMaxScaler
from xgboost import DMatrix
import xgboost
from data_util_2 import *
from omegaconf import OmegaConf

import models

conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_features(pros,labels,infos):
    pros = np.array(pros)  # Convert list of arrays to a 3D numpy array
    print("pros shape:", pros.shape)

    pros=pros.transpose([1,0,2])
    avg_p_diff=calculate_avg_pro_diff(pros)
    avg_info=calculate_avg_info(infos)
    std_info=calculate_std_info(infos)
    std_label=calculate_label_std(labels)
    max_diff_num=get_num_of_most_diff_class(labels)
    feature=np.column_stack((
        std_label,
        avg_info,
        std_info,
        max_diff_num,
        avg_p_diff
    ))
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    return feature
def calculate_info_entropy(pros):
    entropies = []
    for pro in pros:
        # Clip probabilities to avoid log2(0) issues
        pro = np.clip(pro, 1e-12, 1.0)
        entropy = -np.sum(pro * np.log2(pro))
        entropies.append(entropy)
    return entropies

def test(net,testloader):
    net.eval()
    correct = 0
    total = 0
    pros=[]
    labels=[]
    infos=[]
    error_index=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            pro = F.softmax(outputs, dim=1).cpu().numpy()
            pros.append(pro)
            info=calculate_info_entropy(pro)
            infos.extend(info)
            _, predicted = outputs.max(1)
            labels.extend(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            incorrect_mask = ~predicted.eq(targets)
            if incorrect_mask.any():
                incorrect_indices = (batch_idx * testloader.batch_size) + torch.nonzero(incorrect_mask).view(-1)
                error_index.extend(incorrect_indices.tolist())
        acc = 100. * correct / total
    return pros,labels,infos,error_index

def train(net, epoch, optimizer, criterion, trainloader):
    snapshot_root = Path("snapshots") / Path(str(conf.model))
    if not snapshot_root.exists():
        snapshot_root.mkdir(parents=True)
    for epoch_ in range(epoch):
        print('\nEpoch: %d' % epoch_)
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(), snapshot_root / Path('epoch_' + str(epoch_) + '.pth'))


def main():
    net=models.__dict__[conf.model]().to(device)
    trainloader=get_train_data(conf.dataset)
    if conf.dataset=="cifar10" or conf.dataset=="imagenet":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),weight_decay=5e-4,momentum=0.9,lr=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    train(net,conf.epochs,optimizer,criterion,trainloader)
    snapshot_root=Path("snapshots")/Path(conf.model)
    valloader, testloader = get_val_and_test(conf.corruption)
    val_error_index=None
    test_error_index=None
    val_pros=[]
    val_labels=[]
    val_infos=[]
    test_pros=[]
    test_labels=[]
    test_infos=[]
    for epoch in range(conf.epochs):
        net=models.__dict__[conf.model]().to(device)
        net.load_state_dict(torch.load(snapshot_root/Path("epoch_"+str(epoch)+str(".pth"))))
        val_pro,val_label,val_info,val_error_index=test(net,valloader)
        val_pros.extend(val_pro)
        val_labels.extend(val_label)
        val_infos.extend(val_info)
        test_pro,test_label,test_info,test_error_index=test(net,testloader)
        test_pros.extend(test_pros)
        test_labels.extend(test_label)
        test_infos.extend(test_info)

    val_features=extract_features(val_pros,val_labels,val_infos)
    test_features=extract_features(test_pros,test_labels,test_infos)
    val_labels=np.zeros(len(valloader.dataset),dtype=int)
    val_labels[val_error_index]=1

    xgb_rank_params={
        'objective': 'rank:pairwise',
        'colsample_bytree': 0.5,  # This is the ratio of the number of columns used
        'nthread': -1,
        'eval_metric': 'ndcg',
        'max_depth': 5,
        'min_child_weight': 1,
        # 'subsample': 0.6,
        'learning_rate': 0.05,
        # 'n_estimators':50,
        # 'gamma':0.01,
    }
    train_data = DMatrix(val_features, label=val_label)
    rankModel = xgboost.train(xgb_rank_params, train_data)

    test_data=DMatrix(test_features)
    scores=rankModel.predict(test_data)

    test_num=len(testloader.dataset)
    is_bug=np.zeros(test_num)
    is_bug[test_error_index]=1
    index=np.argsort(scores)[::-1]
    is_bug=is_bug[index]
    print(rauc(is_bug,100))
    print(rauc(is_bug,200))
    print(rauc(is_bug,500))
    print(rauc(is_bug,1000))
    print(rauc(is_bug,test_num))
    print(ATRC(is_bug,len(test_error_index)))

if __name__=='__main__':
    main()
