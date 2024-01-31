import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from Predict import ImagePredictor
from PyQt5.QtCore import QThread, pyqtSignal
from main import prepare_data_loaders
from ModelTrainer import ChestXRayModelTrainer

class TrainingThread(QThread):
    training_complete = pyqtSignal(object)  # training_complete sinyali şimdi bir obje gönderecek
    progress_updated = pyqtSignal(int)  # Yeni sinyal

    def __init__(self, train_dir, val_dir, test_dir):
        QThread.__init__(self)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

    def run(self):
        # Veri yükleyicilerini ve ModelTrainer'ı oluşturun
        train_loader, val_loader, test_loader = prepare_data_loaders(self.train_dir, self.val_dir, self.test_dir)
        model_trainer = ChestXRayModelTrainer(train_loader, val_loader)

        # train fonksiyonunu progress_callback ile çağırın
        trained_model, train_losses, val_accuracies = model_trainer.train(
            num_epochs=10,
            progress_callback=self.progress_updated.emit  # Bu, progress_updated sinyalini doğrudan çağıracaktır.
        )

        # Eğitim sonuçlarını sinyalle gönder
        self.training_complete.emit((trained_model, train_losses, val_accuracies))