import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, norm_adj_matrix, dropout=0.2):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.norm_adj_matrix = norm_adj_matrix
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = F.dropout(all_embeddings, p=self.dropout, training=self.training)
        embeddings_list = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
        return final_embeddings[:self.num_users], final_embeddings[self.num_users:]

def bpr_loss(u_emb, p_emb, n_emb, reg_lambda):
    pos_scores = torch.sum(u_emb * p_emb, dim=1)
    neg_scores = torch.sum(u_emb * n_emb, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    reg_loss = (1/2)*(u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)) / u_emb.size(0)
    return loss + reg_lambda * reg_loss
