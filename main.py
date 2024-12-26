import torch
from torch.optim import Adam
from data_processing import load_data, Dianxian
from models import GAT
from train_test import train, test
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
torch.manual_seed(SEED)

# Load Data
fdata, ddata, t1, labels, excel = load_data(SEED)
dataset = Dianxian(fdata, ddata, t1, labels, excel)

# Define Model
model = GAT(90, 90, dropout=0.5, alpha=0.2).to(DEVICE)
optimizer = Adam(model.parameters(), lr=3e-4)

# K-Fold Cross Validation
kfold = KFold(n_splits=4, shuffle=True, random_state=SEED)
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}")
    train_loader = DataLoader(
        dataset, batch_size=1, sampler=SubsetRandomSampler(train_idx)
    )
    test_loader = DataLoader(
        dataset, batch_size=1, sampler=SubsetRandomSampler(test_idx)
    )
    for epoch in range(10):  # Adjust epochs as needed
        train_loss, train_acc = train(model, train_loader, optimizer, DEVICE)
        eval_loss, eval_acc = test(model, test_loader, DEVICE)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Eval Acc: {eval_acc:.4f}"
        )