import numpy as np
import torch

def evaluate_on_df(model, eval_df, K=20, device='cpu'):
    """
    Computes HR@K and NDCG@K over the provided dataframe (can be validation or test).
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()

    user_group = eval_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    HR, NDCG = [], []

    for user, gt_items in user_group.items():
        user_vec = user_emb[user].unsqueeze(0)
        scores = torch.matmul(user_vec.to(device), item_emb.t().to(device)).cpu().numpy().flatten()
        top_K_items = np.argsort(-scores)[:K]

        hits = any(item in gt_items for item in top_K_items)
        dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(top_K_items) if item in gt_items)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt_items), K)))
        ndcg = dcg / idcg if idcg > 0 else 0

        HR.append(1 if hits else 0)
        NDCG.append(ndcg)

    return np.mean(HR), np.mean(NDCG)
