from sentence_transformers import losses

loss_registry = {"cosine": losses.CosineSimilarityLoss, "triplet": losses.TripletLoss}
