# Training or Fine Tuning a sentence transformer model with MNR Loss
from sentence_transformers import SentenceTransformer, losses, InputExample, util
from contrastive_learning.model.losses import loss_registry
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import json
import torch
from pathlib import Path

# where all-MiniLM-L6-v2 is a pre-trained  sentence-transformer model
config = OmegaConf.load(
    "/data/ubuntu/constrastive_representation_learning/src/contrastive_learning/model/config.yaml"
)

model = SentenceTransformer(config.model_name)
with open(config.data_path) as f:
    data_train = json.load(f)
train_examples = [
    InputExample(texts=[instance["query"], instance["pos"], instance["negative"]])
    for instance in data_train
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
train_loss = loss_registry[config.loss_name](model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=config.epochs,
    checkpoint_save_steps=config.checkpoint_save_steps,
    checkpoint_path=config.checkpoint_path,
)
checkpoint = {
    "model_state_dict": model.state_dict(),
}
filename = Path(config.checkpoint_path) / "checkpoint.pth"
torch.save(checkpoint, filename)
