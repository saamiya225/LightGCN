import pandas as pd
import numpy as np
import torch
from scipy.sparse import coo_matrix


def load_dataset(dataset_dir):
    """
    Loads the dataset from the given directory.
    Handles user-item adjacency list format for both train.txt and test.txt.
    """
    # Load user and item ID remaps
    user_list_file = f"{dataset_dir}/user_list.txt"
    item_list_file = f"{dataset_dir}/item_list.txt"

    user_map = pd.read_csv(user_list_file, sep=" ", header=None, names=["org_id", "remap_id"], index_col=0)["remap_id"].to_dict()
    item_map = pd.read_csv(item_list_file, sep=" ", header=None, names=["org_id", "remap_id"], index_col=0)["remap_id"].to_dict()

    # Build train_df from adjacency list format
    train_file = f"{dataset_dir}/train.txt"
    train_user_indices, train_item_indices = [], []
    with open(train_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            user = int(tokens[0])
            items = map(int, tokens[1:])
            for item in items:
                train_user_indices.append(user)
                train_item_indices.append(item)
    train_df = pd.DataFrame({"user_idx": train_user_indices, "item_idx": train_item_indices})

    # Build test_df from adjacency list format
    test_file = f"{dataset_dir}/test.txt"
    test_user_indices, test_item_indices = [], []
    with open(test_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            user = int(tokens[0])
            items = map(int, tokens[1:])
            for item in items:
                test_user_indices.append(user)
                test_item_indices.append(item)
    test_df = pd.DataFrame({"user_idx": test_user_indices, "item_idx": test_item_indices})

    num_users = len(user_map)
    num_items = len(item_map)

    return train_df, test_df, num_users, num_items

def build_norm_adj_matrix(train_df, num_users, num_items):
    user_indices = train_df['user_idx'].values
    item_indices = train_df['item_idx'].values + num_users

    rows = np.concatenate([user_indices, item_indices])
    cols = np.concatenate([item_indices, user_indices])

    adjacency = coo_matrix((np.ones_like(rows), (rows, cols)),
                           shape=(num_users + num_items, num_users + num_items))

    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5, where=degree!=0)
    D_inv_sqrt = coo_matrix((degree_inv_sqrt, (np.arange(len(degree)), np.arange(len(degree)))), 
                            shape=adjacency.shape)
    norm_adj = D_inv_sqrt @ adjacency @ D_inv_sqrt

    norm_adj = norm_adj.tocoo()
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data).float()  
    shape = torch.Size(norm_adj.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
