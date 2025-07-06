import argparse
import time
import torch
import torch.optim as optim
from lightgcn.lightgcn_model import LightGCN, bpr_loss
from lightgcn.utils import sample_train_batch, evaluate_on_df
from lightgcn.dataloader import load_dataset, build_norm_adj_matrix
from lightgcn.evaluate import evaluate_on_df


import numpy as np

# Set all CPU threads on your Mac M2
torch.set_num_threads(8)
print(f" Using {torch.get_num_threads()} CPU threads")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-book', help='dataset name')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
args = parser.parse_args()

print(f" Loading dataset: {args.dataset}")
train_df, test_df, num_users, num_items = load_dataset(f"./datasets/{args.dataset}")
print(f" Train interactions: {len(train_df)}  | Test interactions: {len(test_df)}")

# Device selection
if args.device == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print(" CUDA requested but not available. Using CPU instead.")
        device = torch.device("cpu")
elif args.device == "mps":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print(" MPS requested but not available. Using CPU instead.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f" Using device: {device}")

#  Build normalized adjacency matrix
norm_adj_matrix = build_norm_adj_matrix(train_df, num_users, num_items)
norm_adj_matrix = norm_adj_matrix.to(device)

#  Split test into val + final test
val_frac = 0.2
val_mask = np.random.rand(len(test_df)) < val_frac
val_df = test_df[val_mask]
final_test_df = test_df[~val_mask]
print(f" Validation: {len(val_df)} | Final test: {len(final_test_df)}")

#  Optuna-tuned hyperparameters
embedding_dim = 64
num_layers = 3
reg_lambda = 1.0259391362164704e-05
lr = 0.004887702327256174
batch_size = 1024
epochs = args.epochs
steps = 260

#  Build model & optimizer
model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj_matrix).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

#  Start training
print(" Starting training...")
for epoch in range(1, epochs + 1):
    start_time = time.time()
    model.train()
    total_loss = 0

    for step in range(steps):
        users, pos_items, neg_items = sample_train_batch(train_df, num_users, num_items, batch_size)
        users = torch.LongTensor(users).to(device)
        pos_items = torch.LongTensor(pos_items).to(device)
        neg_items = torch.LongTensor(neg_items).to(device)

        user_emb, item_emb = model()
        u_emb = user_emb[users]
        p_emb = item_emb[pos_items]
        n_emb = item_emb[neg_items]

        loss = bpr_loss(u_emb, p_emb, n_emb, reg_lambda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / steps
    duration = time.time() - start_time
    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f} (took {duration:.2f}s)")

    if epoch % 10 == 0:
        hr, ndcg = evaluate_on_df(model, val_df, K=20)
        print(f"  ↳ Validation HR@20 = {hr:.4f}, NDCG@20 = {ndcg:.4f}")

#  Final test evaluation
test_hr, test_ndcg = evaluate_on_df(model, final_test_df, K=20)
print(f"\n FINAL TEST RESULTS on {args.dataset}: HR@20 = {test_hr:.4f}, NDCG@20 = {test_ndcg:.4f}")

# Save the model
torch.save(model.state_dict(), f"{args.dataset}_lightgcn_final.pt")
print(f"Saved model as {args.dataset}_lightgcn_final.pt")
