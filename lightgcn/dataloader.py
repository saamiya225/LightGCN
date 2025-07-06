import numpy as np
import pandas as pd
import torch

def load_interaction_file_remapped(file_path):
    interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            user = int(tokens[0])
            items = [int(item) for item in tokens[1:]]
            for item in items:
                interactions.append((user, item))
    return pd.DataFrame(interactions, columns=['user_idx', 'item_idx'])

def load_dataset(dataset_path):
    train_file = f"{dataset_path}/train.txt"
    test_file = f"{dataset_path}/test.txt"
    train_df = load_interaction_file_remapped(train_file)
    test_df = load_interaction_file_remapped(test_file)
    num_users = max(train_df['user_idx'].max(), test_df['user_idx'].max()) + 1
    num_items = max(train_df['item_idx'].max(), test_df['item_idx'].max()) + 1
    return train_df, test_df, num_users, num_items

def build_norm_adj_matrix(train_df, num_users, num_items):
    import scipy.sparse as sp
    user_indices = train_df['user_idx'].values
    item_indices = train_df['item_idx'].values + num_users

    data = np.ones(len(user_indices) * 2)
    rows = np.concatenate([user_indices, item_indices])
    cols = np.concatenate([item_indices, user_indices])

    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Ensure it's in COO format before accessing .row / .col
    norm_adj = norm_adj.tocoo()
    indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
    values = torch.FloatTensor(norm_adj.data)
    shape = norm_adj.shape

    return torch.sparse_coo_tensor(indices, values, shape)
