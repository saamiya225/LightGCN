import torch
import numpy as np

def bpr_loss(u_emb, p_emb, n_emb, reg_lambda):
    """
    Bayesian Personalized Ranking (BPR) loss with L2 regularization.
    """
    pos_scores = torch.sum(u_emb * p_emb, dim=1)
    neg_scores = torch.sum(u_emb * n_emb, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    
    reg_loss = (u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)) / u_emb.shape[0]
    return loss + reg_lambda * reg_loss

def sample_train_batch(train_df, num_users, num_items, batch_size):
    """
    Samples a batch of (user, pos_item, neg_item) triples for training.
    """
    users = np.random.choice(train_df["user_idx"].unique(), size=batch_size, replace=True)
    pos_items = []
    neg_items = []

    for user in users:
        # sample a positive item
        pos_item = train_df[train_df["user_idx"] == user].sample(1)["item_idx"].values[0]
        
        # sample a negative item
        while True:
            neg_item = np.random.randint(0, num_items)
            if neg_item not in train_df[train_df["user_idx"] == user]["item_idx"].values:
                break

        pos_items.append(pos_item)
        neg_items.append(neg_item)

    return users, np.array(pos_items), np.array(neg_items)
