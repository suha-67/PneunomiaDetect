import torch
import torch.optim as optim
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ChestXRayModelTrainer:
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = self.create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_model(self):
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        return model

    def train(self, num_epochs=10, progress_callback=None):
        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(images).logits
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)

            if progress_callback is not None:
                progress = int((epoch + 1) / num_epochs * 100)
                progress_callback(progress)
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images).logits
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_train_loss = running_loss / len(self.train_loader.dataset)
            val_accuracy = 100 * correct / total

            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        self.save_model("vit_chest_xray_model.pth")

        return self.model, train_losses, val_accuracies


    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).logits
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        return test_accuracy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def get_predictions(self, test_loader):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images).logits
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        return y_true, y_pred
