import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Visualization:
    @staticmethod
    def display_images(folder, num=5):
        pneumonia_imgs = list(folder.glob('PNEUMONIA/*.jpeg'))[:num]
        normal_imgs = list(folder.glob('NORMAL/*.jpeg'))[:num]

        fig, axes = plt.subplots(nrows=2, ncols=num, figsize=(15, 6))

        for i, img_path in enumerate(normal_imgs):
            img = Image.open(img_path)
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title("NORMAL")
            axes[0, i].axis('off')

        for i, img_path in enumerate(pneumonia_imgs):
            img = Image.open(img_path)
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title("PNEUMONIA")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_results(train_losses, val_accuracies):
        # Eğitim sonuçlarının görselleştirilmesi
        plt.figure(figsize=(12, 5))

        # Training Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Validation Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()