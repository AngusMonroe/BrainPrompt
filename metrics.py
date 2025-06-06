import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def sensitivity(scores, targets):
    y_true = targets.cpu().numpy()
    y_pred = scores.cpu().numpy()

    # Convert predicted probabilities to binary predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Compute sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def precision(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return precision_score(y_true, y_pred)


def recall(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return recall_score(y_true, y_pred)


def f1(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return f1_score(y_true, y_pred)


def roc_auc(scores, targets):
    # y_true = np.zeros((targets.size, targets.max() + 1))
    # y_true[np.arange(targets.size), targets] = 1
    # print(y_true)
    y_true = targets
    y_pred = scores.argmax(axis=1)
    #
    # n_classes = targets.max() + 1
    # auc_scores = []
    # for i in range(n_classes):
    #     y_true_class = np.where(targets == i, 1, 0)
    #     y_pred_class = y_pred[:, i]
    #     auc = roc_auc_score(y_true_class, y_pred_class)
    #     auc_scores.append(auc)
    return roc_auc_score(y_true, y_pred)


def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_all_classes(scores, targets):
    predicted_classes = np.argmax(scores, axis=1)
    correct_predictions = (predicted_classes == targets).astype(float)

    # Calculate accuracy for each class
    class_accuracies = []
    unique_classes = np.unique(targets)
    subject_num = []
    for class_label in unique_classes:
        class_mask = (targets == class_label)
        class_total_samples = np.sum(class_mask)
        subject_num.append(class_total_samples)

        if class_total_samples > 0:
            class_correct_predictions = np.sum(correct_predictions[class_mask])
            class_accuracy = class_correct_predictions / class_total_samples
            class_accuracies.append(class_accuracy)
        else:
            # Handle the case where there are no samples for the class
            class_accuracies.append(float('nan'))
    print(subject_num)

    return class_accuracies


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    acc = acc / len(targets)
    return acc


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

  
def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc
