import torch
from omegaconf import OmegaConf

from sentence_transformers import util, SentenceTransformer

import json
from tqdm import tqdm
import logging

log_format = "%(asctime)s : %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")


def compute_acc(model, data_test):
    nb_accurate = 0
    for i in tqdm(range(len(data_test))):
        instance = data_test[i]
        query_embedding = model.encode(instance["query"])
        embeddings = model.encode([instance["pos"], instance["negative"]])

        cos_sim = util.cos_sim(query_embedding, embeddings)
        if cos_sim[0][0] > cos_sim[0][1]:
            nb_accurate += 1

    return nb_accurate / len(data_test)


config = OmegaConf.load(
    "/data/ubuntu/constrastive_representation_learning/src/contrastive_learning/model/config.yaml"
)
with open(config.data_path) as f:
    data_test = json.load(f)

model = SentenceTransformer(config.model_name)
model.eval()
acc_zero_shot = compute_acc(model, data_test)
logging.info(
    "Accuracy Model without finetuning: ",
)
checkpoint = torch.load(
    "/data/ubuntu/constrastive_representation_learning/saved_checkpoints/checkpoint.pth"
)
model.load_state_dict(checkpoint["model_state_dict"])

acc_fine_tune = compute_acc(model, data_test)
logging.info(f"Accuracy Model zero shot vs fine tuned: , {acc_zero_shot}, {acc_fine_tune}")
