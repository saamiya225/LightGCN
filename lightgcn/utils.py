import numpy as np
import torch

def sample_train_batch(df, num_users, num_items, batch_size, num_negatives=5):
    user_pos_dict = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    users, pos_items, neg_items = [], [], []
    for _ in range(batch_size):
        user = np.random.randint(0, num_users)
        while user not in user_pos_dict:
            user = np.random.randint(0, num_users)
        pos_item = np.random.choice(list(user_pos_dict[user]))
        for _ in range(num_negatives):
            neg_item = np.random.randint(0, num_items)
            while neg_item in user_pos_dict[user]:
                neg_item = np.random.randint(0, num_items)
            users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)
    return users, pos_items, neg_items

def evaluate_on_df(model, eval_df, K=20):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()
    user_group = eval_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    HR, NDCG = [], []
    for user, gt_items in user_group.items():
        user_vec = user_emb[user].unsqueeze(0)
        scores = torch.matmul(user_vec, item_emb.t()).cpu().numpy().flatten()
        top_K_items = np.argsort(-scores)[:K]
        hits = any(item in gt_items for item in top_K_items)
        dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(top_K_items) if item in gt_items)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt_items), K)))
        ndcg = dcg / idcg if idcg > 0 else 0
        HR.append(1 if hits else 0)
        NDCG.append(ndcg)
    return np.mean(HR), np.mean(NDCG)
