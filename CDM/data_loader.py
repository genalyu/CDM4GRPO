import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class InteractionDataset(Dataset):
    def __init__(self, user_ids, item_ids, scores, q_matrix):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.scores = scores
        self.q_matrix = q_matrix

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        score = self.scores[idx]

        kn_emb = self.q_matrix[item_id]

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.tensor(kn_emb, dtype=torch.float32),
            torch.tensor(score, dtype=torch.float32),
        )


def load_data_from_csv(interaction_path, q_matrix_path, batch_size=32, val_ratio=0.1):
    interactions = pd.read_csv(interaction_path)
    q_matrix = pd.read_csv(q_matrix_path, index_col=0).values.astype(np.float32)

    user_ids = interactions['user_id'].astype('category').cat.codes.values
    item_ids = interactions['item_id'].astype('category').cat.codes.values
    scores = interactions['score'].values.astype(np.float32)

    num_users = len(np.unique(user_ids))
    num_items = len(np.unique(item_ids))
    num_knowledge = q_matrix.shape[1]

    train_idx, val_idx = train_test_split(np.arange(len(user_ids)), test_size=val_ratio, random_state=42)
    train_dataset = InteractionDataset(user_ids[train_idx], item_ids[train_idx], scores[train_idx], q_matrix)
    val_dataset = InteractionDataset(user_ids[val_idx], item_ids[val_idx], scores[val_idx], q_matrix)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return {
        'user_id': user_ids,
        'item_id': item_ids,
        'score': scores,
        'q_matrix': q_matrix,
        'num_users': num_users,
        'num_items': num_items,
        'num_knowledge': num_knowledge,
        'train_loader': train_loader,
        'val_loader': val_loader
    }
