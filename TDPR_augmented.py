import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from sklearn.preprocessing import MinMaxScaler
from data_util_2 import *  # Ensure get_augmentation_pipeline() and other helpers are defined here.
from omegaconf import OmegaConf
from BugNet import *
import models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
import numpy as np

def plot_trc(trc_values, label='Model', color='blue'):
    ranks = np.arange(1, len(trc_values) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, trc_values, marker='o', markersize=1, linestyle='-', color=color, label=label)
    plt.xlabel('Rank')
    plt.ylabel('TRC Value')
    plt.title('TRC Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"trc_curve2_{label}.png", dpi=300, bbox_inches='tight')
    plt.show()



def debug_features(features, labels):
    df = pd.DataFrame(features, columns=[
        'std_label', 
        'avg_info', 
        # 'std_info',
        'max_diff', 
        # 'avg_pro_diff',
        # 'kl_div', 
        'agreement', 
        'margin', 
        'mutual_info'
    ])
    df['label'] = labels

    # Variance
    print("\n🔍 Feature Variance:")
    print(df.var())

    # Correlation heatmap
    print("\n📈 Correlation Matrix:")
    corr = df.drop(columns='label').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Class-wise mean
    print("\n📊 Class-wise Means:")
    print(df.groupby('label').mean())


conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Augmentation and Forward Pass Functions ---
def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)


def forward_with_augmentations(net, sample, num_aug=conf.augs):
    if isinstance(sample, torch.Tensor):
        sample = transforms.ToPILImage()(sample.cpu())
    aug_pipeline = get_augmentation_pipeline()  
    prob_list, label_list, uncertainty_list = [], [], []
    net.eval()
    with torch.no_grad():
        for _ in range(num_aug):
            aug_sample = aug_pipeline(sample).unsqueeze(0).to(device)
            outputs = net(aug_sample)
            probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze(0)
            prob_list.append(probs)
            label_list.append(np.argmax(probs))
            uncertainty_list.append(calculate_info_entropy_from_probs(probs))
    return np.array(prob_list), np.array(label_list), np.array(uncertainty_list)

def generate_augmented_outputs(net, dataset, num_aug=conf.augs):
    num_samples = len(dataset)
    first_sample, _ = dataset[0]
    if isinstance(first_sample, torch.Tensor):
        first_sample = first_sample.to(device)
    with torch.no_grad():
        num_classes = net(first_sample.unsqueeze(0)).shape[1]
    
    all_prob_arrays = np.zeros((num_samples, num_aug, num_classes))
    all_label_arrays = np.zeros((num_samples, num_aug))
    all_uncertainty_arrays = np.zeros((num_samples, num_aug))
    
    for idx in range(num_samples):
        sample, _ = dataset[idx]
        probs, labels, uncertainties = forward_with_augmentations(net, sample, num_aug=num_aug)
        all_prob_arrays[idx] = probs
        all_label_arrays[idx] = labels
        all_uncertainty_arrays[idx] = uncertainties
    return all_prob_arrays, all_label_arrays, all_uncertainty_arrays

# --- Helper Functions for Feature Extraction ---
def calculate_avg_pro_diff(pros):
    from sklearn.metrics.pairwise import cosine_similarity
    num_samples, num_aug, _ = pros.shape
    avg_diffs = np.zeros(num_samples)
    for i in range(num_samples):
        ref = pros[i, -1, :].reshape(1, -1)
        sims = cosine_similarity(pros[i, :num_aug-1, :], ref)
        distances = 1 - sims.flatten()
        avg_diffs[i] = np.mean(distances)
    return avg_diffs

def get_num_of_most_diff_class(labels):
    num_samples, num_aug = labels.shape
    max_diff = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        target = labels[i, -1]
        diff_counts = {}
        for j in range(num_aug - 1):
            if labels[i, j] != target:
                diff_counts[labels[i, j]] = diff_counts.get(labels[i, j], 0) + 1
        max_diff[i] = max(diff_counts.values()) if diff_counts else 0
    return max_diff

def calculate_kl_divergence(pros):
    kl_div = []
    for p in pros:
        base = p[-1]
        kl = [np.sum(p_i * np.log((p_i + 1e-12) / (base + 1e-12))) for p_i in p[:-1]]
        kl_div.append(np.mean(kl))
    return np.array(kl_div)

def calculate_agreement(labels):
    agreement_scores = []
    for row in labels:
        mode_label = np.bincount(row[:-1].astype(int)).argmax()
        agreement = np.sum(row[:-1] == mode_label) / (len(row) - 1)
        agreement_scores.append(agreement)
    return np.array(agreement_scores)

def calculate_margin(pros):
    # Difference between top-1 and top-2 class probabilities
    margins = []
    for p in pros:
        top2 = np.partition(p[-1], -2)[-2:]  # last row is reference
        margins.append(top2[-1] - top2[-2])
    return np.array(margins)

def calculate_mutual_information(pros):
    # MI = H(mean_p) - mean(H(p))
    eps = 1e-12
    mean_p = np.mean(pros, axis=1)
    entropy_mean = -np.sum(mean_p * np.log2(mean_p + eps), axis=1)

    entropy_per = -np.sum(pros * np.log2(pros + eps), axis=2)
    mean_entropy = np.mean(entropy_per, axis=1)

    return entropy_mean - mean_entropy


def extract_features(pros, labels, infos):
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = np.mean(infos, axis=1)
    std_info = np.std(infos, axis=1)
    std_label = np.std(labels, axis=1)
    max_diff_num = get_num_of_most_diff_class(labels)
    kl_divs = calculate_kl_divergence(pros)
    agreements = calculate_agreement(labels)
    margins = calculate_margin(pros)
    mutual_infos = calculate_mutual_information(pros)

    feature = np.column_stack((
        std_label, 
        avg_info, 
        # std_info, 
        max_diff_num, 
        # avg_p_diff, 
        # kl_divs, 
        agreements, 
        margins, 
        mutual_infos
    ))
    
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)


def calculate_info_entropy(pros):
    entropys = []
    for pro in pros:
        entropy = -np.sum(pro * np.log2(pro))
        entropys.append(entropy)
    return entropys

# --- Original Test Function (unchanged) ---
def test(net, testloader):
    net.eval()
    correct, total = 0, 0
    pros, labels, infos, error_index = [], [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            pro = F.softmax(outputs, dim=1).cpu().numpy()
            pros.extend(pro)
            infos.extend(calculate_info_entropy(pro))
            _, predicted = outputs.max(1)
            labels.extend(predicted.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            incorrect_mask = ~predicted.eq(targets)
            if incorrect_mask.any():
                incorrect_indices = (batch_idx * testloader.batch_size) + torch.nonzero(incorrect_mask).view(-1)
                error_index.extend(incorrect_indices.tolist())
    acc = 100. * correct / total
    # print(f"\n🧪 Final Test Accuracy: {acc:.2f}%")
    return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

def train(net, num_epochs, optimizer, criterion, trainloader, device):
    net.to(device)
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    for epoch in range(num_epochs):
        net.train()  # Set model to training mode
        correct = 0
        total = 0
        running_loss = 0.0
        
        print(f"\nEpoch: {epoch + 1}")
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item()
        
        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        scheduler.step()  # Step the scheduler after each epoch

        print(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        

def compute_class_weights(y_train_tensor, device):
    y_train_np = y_train_tensor.cpu().numpy()
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)
    return torch.tensor(weights, dtype=torch.float32).to(device)

from sklearn.model_selection import RandomizedSearchCV

def run_rf_random_search(X_train, y_train, X_test, test_error_index):
    test_flags = np.zeros(len(X_test))
    test_flags[test_error_index] = 1

    param_grid = {
        "n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None],
    }

    clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42)
    rs_clf = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=30,
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    rs_clf.fit(X_train, y_train)

    # Predict probabilities
    scores = rs_clf.predict_proba(X_test)[:, 1]  # probability of class 1 (bug)
    ranking = np.argsort(scores)[::-1]
    sorted_flags = test_flags[ranking]

    rauc_100 = rauc(sorted_flags, 100)
    rauc_200 = rauc(sorted_flags, 200)
    rauc_500 = rauc(sorted_flags, 500)
    rauc_1000 = rauc(sorted_flags, 1000)
    rauc_all = rauc(sorted_flags, len(test_flags))
    atrc_val = ATRC(sorted_flags, int(np.sum(test_flags)))
    print("Best parameters found:", rs_clf.best_params_)
    print("Number of bug samples in top 100:", np.sum(sorted_flags[:100]))
    print("Number of bug samples in top 500:", np.sum(sorted_flags[:500]))


    return rauc_100, rauc_200, rauc_500, rauc_1000, rauc_all, atrc_val


def run_rf_grid(X_train, y_train, X_test, test_error_index):
    test_flags = np.zeros(len(X_test))
    test_flags[test_error_index] = 1

    results = []
    # grid = ParameterGrid({
    #     'n_estimators': [50, 100],
    #     'max_depth': [4, 8, None]
    # })

    # for params in grid:
    model = RandomForestClassifier(n_estimators=50,
                                      max_depth=None,
                                      class_weight='balanced')
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    ranking = np.argsort(scores)[::-1]
    sorted_flags = test_flags[ranking]

    rauc_100 = rauc(sorted_flags, 100)
    rauc_200 = rauc(sorted_flags, 200)
    rauc_500 = rauc(sorted_flags, 500)
    rauc_1000 = rauc(sorted_flags, 1000)
    rauc_all = rauc(sorted_flags, len(test_flags))
    atrc_val, trc_values = ATRC(sorted_flags, int(np.sum(test_flags)))
    plot_trc(trc_values, label='TDPR Model', color='green')
    results.append((rauc_100, rauc_200, rauc_500, rauc_1000, rauc_all, atrc_val))

    return results

# --- Main Function ---
def main():
    # Initialize model and training data.
    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    if conf.dataset in ["cifar10", "imagenet"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
    # Train the model.
    train(net, conf.epochs, optimizer, criterion, trainloader, device)
    # final_model_path = "final_model.pth"
    # torch.save(net.state_dict(), final_model_path)
    # state_dict = torch.load(final_model_path, map_location=device)
    # missing, unexpected = net.load_state_dict(state_dict, strict=False)

    # if missing or unexpected:
    #     print("⚠️  Mismatch while loading weights")
    #     print("   • Missing keys:    ", missing)
    #     print("   • Unexpected keys: ", unexpected)

    # net.eval() 

    # Get validation and test DataLoaders.
    valloader, testloader = get_val_and_test(conf.corruption)
    
    # Set up file paths for saving intermediate arrays.
    aug_file = "augmented_outputs.npz"
    feat_file = "extracted_features.npz"
    err_file = "error_indices.npz"
    
    # If augmented outputs exist, load them; otherwise compute and save.
    if os.path.exists(aug_file):
        data = np.load(aug_file)
        val_prob_arrays = data['val_prob_arrays']
        val_label_arrays = data['val_label_arrays']
        val_uncertainty_arrays = data['val_uncertainty_arrays']
        test_prob_arrays = data['test_prob_arrays']
        test_label_arrays = data['test_label_arrays']
        test_uncertainty_arrays = data['test_uncertainty_arrays']
        # print("Loaded augmented outputs from file.")
    else:
        val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=conf.augs)
        test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=conf.augs)
        np.savez(aug_file,
                 val_prob_arrays=val_prob_arrays,
                 val_label_arrays=val_label_arrays,
                 val_uncertainty_arrays=val_uncertainty_arrays,
                 test_prob_arrays=test_prob_arrays,
                 test_label_arrays=test_label_arrays,
                 test_uncertainty_arrays=test_uncertainty_arrays)
        # print("Computed and saved augmented outputs.")
    
    # If extracted features exist, load them; otherwise compute and save.
    # if os.path.exists(feat_file):
    #     feat_data = np.load(feat_file)
    #     val_features = feat_data['val_features']
    #     test_features = feat_data['test_features']
    #     # print("Loaded extracted features from file.")
    # else:
    val_features = extract_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    test_features = extract_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
        # np.savez(feat_file, val_features=val_features, test_features=test_features)
        # print("Computed and saved extracted features.")
    
    # If error indices exist, load them; otherwise compute and save.
    if os.path.exists(err_file):
        err_data = np.load(err_file)
        val_error_index = err_data['val_error_index']
        test_error_index = err_data['test_error_index']
        # print("Loaded error indices from file.")
    else:
        _, _, _, val_error_index = test(net, valloader)
        _, _, _, test_error_index = test(net, testloader)
        np.savez(err_file, val_error_index=val_error_index, test_error_index=test_error_index)
        # print("Computed and saved error indices.")

    print("Extracted validation features shape:", val_features.shape)
    print("Extracted test features shape:", test_features.shape)
    
    # Create binary labels: 0 for correct predictions, 1 for bug-revealing samples.
    val_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_labels[val_error_index] = 1

    X_vis = PCA(n_components=2).fit_transform(val_features)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=val_labels, cmap='coolwarm')
    plt.title("PCA of Validation Features")
    plt.savefig("pca_val_features.png", dpi=300, bbox_inches='tight')

    X_vis = TSNE(n_components=2, random_state=42).fit_transform(val_features)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=val_labels, cmap='coolwarm')
    plt.title("t-SNE of Validation Features")
    plt.savefig("tsne_val_features.png")
    
    # Helps you see: are low-confidence predictions scattered randomly, or localized?
    confidences = np.max(val_prob_arrays[:, -1, :], axis=1)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=confidences, cmap='viridis')
    plt.colorbar(label='Model Confidence')
    plt.title("Confidence Distribution in Feature Space")
    plt.savefig("confidence_distribution.png", dpi=300, bbox_inches='tight')
    
    
    print("val prob arrays",val_prob_arrays[:5])
    print("val label arrays",val_label_arrays[:20])
    print("val uncertainty arrays",val_uncertainty_arrays[:5])
    print("Validation labels:", val_labels)
    print("Validation error indices:", val_error_index)
    print("Test error indices:", test_error_index)
    print("Validation features shape:", val_features.shape)
    print("Test features shape:", test_features.shape)
    print(val_features[:5])
    print(val_labels[:5])

    debug_features(val_features, val_labels)

    # Hyperparameter for hidden layer size
    hidden_dim = 64  # You can make this a configurable argument

    # Convert data to PyTorch tensors
    X_train = torch.tensor(val_features, dtype=torch.float32)
    y_train = torch.tensor(val_labels, dtype=torch.float32)
    X_test = torch.tensor(test_features, dtype=torch.float32)

    # Choose model: "rf" or "nn"
    model_type = conf.model_type    

    if model_type == "nn":
        model = BugNet(input_dim=val_features.shape[1], hidden_dim=128).to(device)
        class_weights = compute_class_weights(y_train, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train.to(device))         # Shape: (batch,)
            loss = criterion(outputs, y_train.to(device))  # Both are float, no softmax!
            loss.backward()
            optimizer.step()
            # Compute accuracy
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).long()
            correct = (predicted == y_train.to(device)).sum().item()
            total = y_train.size(0)
            accuracy = 100.0 * correct / total
            # print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%")

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test.to(device)))  # Sigmoid instead of softmax
            scores = probs.cpu().numpy()  # Shape: (batch,)


        test_flags = np.zeros(len(testloader.dataset))
        test_flags[test_error_index] = 1
        index = np.argsort(scores)[::-1]
        sorted_flags = test_flags[index]
        print("RAUC@100:", rauc(sorted_flags, 100))
        print("RAUC@200:", rauc(sorted_flags, 200))
        print("RAUC@500:", rauc(sorted_flags, 500))
        print("RAUC@1000:", rauc(sorted_flags, 1000))
        print("RAUC@all:", rauc(sorted_flags, len(testloader.dataset)))
        print("ATRC:", ATRC(sorted_flags, int(np.sum(test_flags))))

    elif model_type == "rf":
        # r100, r200, r500, r1000, rauc_all, atrc = run_rf_random_search(val_features, val_labels, test_features, test_error_index)
        # print(f"RAUC@100: {r100}")
        # print(f"RAUC@200: {r200}")
        # print(f"RAUC@500: {r500}")
        # print(f"RAUC@1000: {r1000}")
        # print(f"RAUC@all: {rauc_all}")
        # print(f"ATRC: {atrc}")
        results = run_rf_grid(val_features, val_labels, test_features, test_error_index)
        for r100, r200, r500, r1000, rauc_all, atrc in results:
            print(f"RAUC@100: {r100}")
            print(f"RAUC@200: {r200}")
            print(f"RAUC@500: {r500}")
            print(f"RAUC@1000: {r1000}")
            print(f"RAUC@all: {rauc_all}")
            print(f"ATRC: {atrc}")

                  


if __name__ == '__main__':
    main()
