import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from Predict import ImagePredictor
from PyQt5.QtCore import QThread, pyqtSignal

class TestThread(QThread):
    result_ready = pyqtSignal(str)  # İşlem sonucunu iletmek için kullanılacak sinyal

    def __init__(self, image_path, model_path):
        QThread.__init__(self)
        self.image_path = image_path
        self.model_path = model_path

    def run(self):
        # ImagePredictor sınıfı yarat ve resmi test et
        predictor = ImagePredictor(self.model_path)
        prediction = predictor.predict_image(self.image_path)
        self.result_ready.emit(prediction)
