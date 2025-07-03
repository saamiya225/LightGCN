import torch
import argparse
from lightgcn.dataloader import load_dataset
from lightgcn.model import LightGCN
from lightgcn.utils import bpr_loss, sample_train_batch
from lightgcn.evaluate import evaluate_on_df
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-book')
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--reg_lambda', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--steps', type=int, default=260)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# =====================
# Load dataset
# =====================
print(f"📚 Loading dataset: {args.dataset}")
train_df, test_df, num_users, num_items = load_dataset(f"./datasets/{args.dataset}")

# Split test into validation + final test
val_df = test_df.sample(frac=0.5, random_state=42)
final_test_df = test_df.drop(val_df.index)
print(f"✅ Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(final_test_df)}")

# =====================
# Build normalized adjacency matrix
# =====================
interactions = pd.concat([train_df[["user_idx", "item_idx"]], val_df[["user_idx", "item_idx"]]])
interactions["item_idx"] += num_users  # shift item indices
rows, cols = interactions["user_idx"], interactions["item_idx"]
values = np.ones(len(interactions))
adj = coo_matrix((values, (rows, cols)), shape=(num_users + num_items, num_users + num_items))
adj = adj + adj.T  # symmetric
deg = np.array(adj.sum(axis=1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5, where=deg!=0)
D_inv_sqrt = coo_matrix((deg_inv_sqrt, (np.arange(len(deg_inv_sqrt)), np.arange(len(deg_inv_sqrt)))), shape=adj.shape)
norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
norm_adj = norm_adj.tocoo()
indices = torch.tensor([norm_adj.row, norm_adj.col], dtype=torch.long)
values = torch.tensor(norm_adj.data, dtype=torch.float32)
norm_adj_matrix = torch.sparse.FloatTensor(indices, values, torch.Size(norm_adj.shape)).to(args.device)

# =====================
# Instantiate model
# =====================
model = LightGCN(num_users, num_items, args.embedding_dim, args.num_layers, norm_adj_matrix).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# =====================
# Training loop
# =====================
print("🚀 Starting training...")
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0

    for step in range(args.steps):
        users, pos_items, neg_items = sample_train_batch(train_df, num_users, num_items, args.batch_size)
        users = torch.LongTensor(users).to(args.device)
        pos_items = torch.LongTensor(pos_items).to(args.device)
        neg_items = torch.LongTensor(neg_items).to(args.device)

        user_emb, item_emb = model()
        u_emb = user_emb[users]
        p_emb = item_emb[pos_items]
        n_emb = item_emb[neg_items]

        loss = bpr_loss(u_emb, p_emb, n_emb, args.reg_lambda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / args.steps
    print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.6f}")

    if epoch % 10 == 0:
        hr, ndcg = evaluate_on_df(model, val_df, K=20, device=args.device)
        print(f"  ↳ Validation HR@20 = {hr:.4f}, NDCG@20 = {ndcg:.4f}")

# =====================
# Final evaluation on test
# =====================
test_hr, test_ndcg = evaluate_on_df(model, final_test_df, K=20, device=args.device)
print(f"\n🎯 FINAL TEST RESULTS on {args.dataset}: HR@20 = {test_hr:.4f}, NDCG@20 = {test_ndcg:.4f}")
