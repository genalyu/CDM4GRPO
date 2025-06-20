# === train.py ===
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from .data_loader import load_data_from_csv
from .model import Net
import sys
from sklearn.metrics import mean_absolute_error, r2_score
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--interaction_csv', type=str, default='CDM/interaction.csv', help='Path to interaction.csv')
parser.add_argument('--q_matrix_csv', type=str, default='CDM/q_matrix.csv', help='Path to q_matrix.csv')
parser.add_argument('--device', type=str, default=device, help='Device to use, e.g., cuda:0 or cpu')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')

args = parser.parse_args()

# Set device and epochs
device = torch.device(args.device)
epoch_n = args.epochs


# Load data
if args.interaction_csv and args.q_matrix_csv:
    data = load_data_from_csv(args.interaction_csv, args.q_matrix_csv, batch_size=64)
    q_matrix = data['q_matrix']
    student_n = data['num_users']
    exer_n = data['num_items']
    knowledge_n = data['num_knowledge']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
else:
    raise ValueError("CSV data paths must be provided via --interaction_csv and --q_matrix_csv")

# Model initialization
net = Net(student_n, exer_n, knowledge_n, q_matrix).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.002)
loss_function = nn.MSELoss()

# Training loop
def train():
    print('training model...')
    for epoch in range(epoch_n):
        net.train()
        running_loss = 0.0
        for batch_idx, (stu_ids, exer_ids, kn_embs, labels) in enumerate(train_loader):
            stu_ids, exer_ids, kn_embs, labels = stu_ids.to(device), exer_ids.to(device), kn_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(stu_ids, exer_ids, kn_embs)
            loss = loss_function(output, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            with torch.no_grad():
                net.k_difficulty.weight.data *= torch.tensor(data['q_matrix'], dtype=torch.float32).to(device)

            running_loss += loss.item()
        
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0
        rmse, mae, r2 = validate(net, epoch)
        save_snapshot(net, f'model/model_epoch{epoch+1}')

def validate(model, epoch):
    print('validating model...')
    net.eval()

    pred_all, label_all = [], []
    with torch.no_grad():
        for stu_ids, exer_ids, kn_embs, labels in val_loader:
            stu_ids, exer_ids, kn_embs, labels = stu_ids.to(device), exer_ids.to(device), kn_embs.to(device), labels.to(device)
            output = model(stu_ids, exer_ids, kn_embs)
            output = output.view(-1)
            pred_all += output.cpu().tolist()
            label_all += labels.cpu().tolist()
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    mae = mean_absolute_error(label_all, pred_all)
    r2 = r2_score(label_all, pred_all)
    print('epoch= %d, rmse= %f, mae= %f, r2= %f' % (epoch+1, rmse, mae, r2))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, rmse= %f, mae= %f, r2= %f\n' % (epoch+1, rmse, mae, r2))

    return rmse, mae, r2
# Save model
def save_snapshot(model, filename):
    torch.save(model.state_dict(), filename)

# Start training
if __name__ == '__main__':
    train()

class StudentEmbeddingOptimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

        for param in self.model.parameters():
            param.requires_grad = False

    def optimize_embedding(self, exer_id, kn_emb, target_output, steps=5):
        device = "cuda:0"
        exer_id = exer_id.to(device)
        kn_emb = kn_emb.to(device)
        target_output = target_output.to(device)

        student_emb = torch.randn((1, self.model.knowledge_dim), requires_grad=True, device=device)

        optimizer = torch.optim.Adam([student_emb], lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(steps):
            optimizer.zero_grad()

            pred = self.model.forward(
                stu_id=None,  
                exer_id=exer_id.unsqueeze(0),  
                kn_emb=kn_emb.unsqueeze(0),  
                student_emb=student_emb  
            )

            loss = loss_fn(pred, target_output)

            loss.backward()
            optimizer.step()


        return student_emb.detach().squeeze(0)