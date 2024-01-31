import torch
from transformers import ViTForImageClassification, ViTConfig


class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        return self.model