import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold


def train(model, datasets_train, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for data in datasets_train:
        inputs = [d.to(device) for d in data]
        node, edge, t1, label, excel = inputs
        out = model(node.float(), edge.float())
        loss = F.nll_loss(out, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (out.argmax(1) == label).sum().item() / len(label)
    return train_loss / len(datasets_train), train_acc / len(datasets_train)


def test(model, datasets_test, device):
    model.eval()
    with torch.no_grad():
        eval_loss = 0
        labels_all, preds_all = [], []
        for data in datasets_test:
            inputs = [d.to(device) for d in data]
            node, edge, t1, label, excel = inputs
            out = model(node.float(), edge.float())
            loss = F.nll_loss(out, label.long())
            eval_loss += loss.item()
            labels_all.extend(label.cpu().numpy())
            preds_all.extend(out.argmax(1).cpu().numpy())

        accuracy = accuracy_score(labels_all, preds_all)
        return eval_loss / len(datasets_test), accuracy