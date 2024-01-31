from transformers import ViTForImageClassification, ViTConfig
import torch
from torchvision import transforms
from PIL import Image

class ImagePredictor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_model(self, model_path):
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image).logits
            _, predicted = torch.max(outputs, 1)

        return "NORMAL" if predicted[0] == 0 else "PNEUMONIA"