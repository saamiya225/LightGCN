import pandas as pd

def load_dataset(dataset_dir):
    """
    Loads the dataset from the given directory.
    Expects train.txt, test.txt, user_list.txt, item_list.txt.
    Returns train_df, test_df, num_users, num_items.
    """

    # Load user and item ID remaps
    user_list_file = f"{dataset_dir}/user_list.txt"
    item_list_file = f"{dataset_dir}/item_list.txt"

    user_map = pd.read_csv(user_list_file, sep=" ", header=None, names=["org_id", "remap_id"], index_col=0)["remap_id"].to_dict()
    item_map = pd.read_csv(item_list_file, sep=" ", header=None, names=["org_id", "remap_id"], index_col=0)["remap_id"].to_dict()

    # Load train and test interactions
    train_file = f"{dataset_dir}/train.txt"
    test_file = f"{dataset_dir}/test.txt"

    train_df = pd.read_csv(train_file, sep=" ", header=None, names=["user", "item"])
    test_df = pd.read_csv(test_file, sep=" ", header=None, names=["user", "item"])

    # Map to integer indices
    train_df["user_idx"] = train_df["user"].map(user_map)
    train_df["item_idx"] = train_df["item"].map(item_map)
    test_df["user_idx"] = test_df["user"].map(user_map)
    test_df["item_idx"] = test_df["item"].map(item_map)

    num_users = len(user_map)
    num_items = len(item_map)

    return train_df, test_df, num_users, num_items
