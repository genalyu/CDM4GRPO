import torch
import pandas as pd
import numpy as np
from model import Net

q_matrix = pd.read_csv("q_matrix.csv", index_col=0).values.astype(np.float32)
model = Net(7, 1500, 8, q_matrix)  
model.load_state_dict(torch.load("model/model_epoch3"))  

exer_ids = torch.arange(1500)
k_difficulty = model.k_difficulty(exer_ids).detach().cpu().numpy()

k_min = k_difficulty.min()
k_max = k_difficulty.max()
k_difficulty = (k_difficulty - k_min) / (k_max - k_min)

k_difficulty *= q_matrix

pd.DataFrame(k_difficulty).to_csv("k_difficulty.csv", index=False)