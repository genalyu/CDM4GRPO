import pandas as pd
from datasets import load_dataset

d1 = load_dataset("genalyu/gemini-2.0-flash-1500samples-with-scores", split="train")
d2 = load_dataset("genalyu/sillyECNU-lite-1500samples-with-scores", split="train")
d3 = load_dataset("genalyu/ECNU-lite-1500samples-with-scores", split="train")
d4 = load_dataset("genalyu/ECNU-plus-1500samples-with-scores", split="train")
d5 = load_dataset("genalyu/gemini-1.5-flash-8B-1500samples-with-scores", split="train")
d6 = load_dataset("genalyu/Openr1_1500samples_with_scores", split="train")
d7 = load_dataset("genalyu/gemini-1.5-flash-1500samples-with-scores", split="train")

d1= pd.DataFrame(d1)
d2= pd.DataFrame(d2)
d3= pd.DataFrame(d3)
d4= pd.DataFrame(d4)
d5= pd.DataFrame(d5)
d6= pd.DataFrame(d6)
d7= pd.DataFrame(d7)

d1["user_id"] = "d1"
d2["user_id"] = "d2"
d3["user_id"] = "d3"
d4["user_id"] = "d4"
d5["user_id"] = "d5"
d6["user_id"] = "d6"
d7["user_id"] = "d7"

d1["score"] = ((d1["reward_score"].astype(float)) + 2) / 9
d2["score"] = ((d2["reward_score"].astype(float)) + 2) / 9
d3["score"] = ((d3["reward_score"].astype(float)) + 2) / 9
d4["score"] = ((d4["reward_score"].astype(float)) + 2) / 9
d5["score"] = ((d5["reward_score"].astype(float)) + 2) / 9
d6["score"] = ((d6["reward_score"].astype(float)) + 2) / 9
d7["score"] = ((d7["reward_score"].astype(float)) + 2) / 9

d1["item_id"] = d1.index
d2["item_id"] = d2.index
d3["item_id"] = d3.index
d4["item_id"] = d4.index
d5["item_id"] = d5.index
d6["item_id"] = d6.index
d7["item_id"] = d7.index

interactions = pd.concat([
    d1[["user_id", "item_id", "score"]],
    d2[["user_id", "item_id", "score"]],
    d3[["user_id", "item_id", "score"]],
    d4[["user_id", "item_id", "score"]],
    d5[["user_id", "item_id", "score"]],
    d6[["user_id", "item_id", "score"]],
    d7[["user_id", "item_id", "score"]]
], ignore_index=True)
interactions.to_csv("interaction.csv", index=False)

def parse_knowledge_tags(tag_str):
    return tag_str.split('+')

all_tags = sorted(set(
    tag for row in d6["problem_type"].tolist() 
    for tag in parse_knowledge_tags(row)
))

# 构建 q_matrix.csv
def one_hot(tags, tag_list):
    tag_set = set(tags.split('+'))
    return [1 if tag in tag_set else 0 for tag in tag_list]

q_matrix = pd.DataFrame(
    [one_hot(tags, all_tags) for tags in d6["problem_type"]],
    columns=all_tags
)
q_matrix.insert(0, "item_id", d1["item_id"])
q_matrix.to_csv("q_matrix.csv", index=False)