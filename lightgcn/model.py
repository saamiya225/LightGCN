import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, norm_adj_matrix):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.norm_adj_matrix = norm_adj_matrix

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self):
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]

        # Perform propagation for num_layers
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Average embeddings across layers
        final_embedding = torch.stack(embeddings_list, dim=1).mean(dim=1)
        user_emb = final_embedding[:self.num_users]
        item_emb = final_embedding[self.num_users:]

        return user_emb, item_emb
