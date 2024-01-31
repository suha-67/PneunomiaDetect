import sys
import main
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QGroupBox, QProgressBar
from TrainingThread import TrainingThread
from TestThread import TestThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal


class ChestXRayApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chest X-Ray Analysis")
        self.setGeometry(100, 100, 600, 400)
        self.progress_label = QLabel("Eğitim durumu bekleniyor...")
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(200, 200)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        training_group = QGroupBox("Eğitim")
        training_layout = QVBoxLayout()
        self.train_button = QPushButton("Eğitim Başlat")
        self.train_button.clicked.connect(self.train_model)
        training_layout.addWidget(self.train_button)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)  # Maksimum değer
        self.progress_bar.setAlignment(Qt.AlignCenter)
        training_layout.addWidget(self.progress_label)
        training_layout.addWidget(self.progress_bar)
        training_group.setLayout(training_layout)
        training_group.setFixedHeight(100)
        layout.addWidget(training_group)
        image_processing_group = QGroupBox("Görüntü İşleme")
        image_processing_layout = QVBoxLayout()
        self.test_button = QPushButton("Görüntü Seç ve Test Et")
        self.test_button.clicked.connect(self.select_image)
        image_processing_layout.addWidget(self.test_button)
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)  # İçeriği ortalar
        pixmap = QPixmap("default/defaultXray.png")  # Varsayılan resmi yükler
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Resmi ölçeklendir
        self.image_label.setPixmap(pixmap)
        self.image_label.clicked.connect(self.select_image)
        image_processing_layout.addWidget(self.image_label)
        self.result_label = QLabel("Sonuçlar burada gösterilecek.")
        image_processing_layout.addWidget(self.result_label)
        image_processing_group.setFixedHeight(400)
        image_processing_group.setLayout(image_processing_layout)
        layout.addWidget(image_processing_group)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def train_model(self):
        self.progress_label.setText("Eğitim Başlatıldı: %0")
        self.progress_bar.setValue(0)

        # TrainingThread oluşturun ve sinyalleri bağlayın
        self.training_thread = TrainingThread(main.train_dir, main.val_dir, main.test_dir)
        self.training_thread.progress_updated.connect(self.update_progress)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.start()

    def update_progress(self, progress):
        if(progress != 100):
            self.progress_label.setText(f"Eğitim Başlatıldı: %{progress}")
        else:
            self.progress_label.setText(f"Eğitim Tamamlandı.")

        self.progress_bar.setValue(progress)
        self.progress_bar.setAlignment(Qt.AlignCenter)


    def on_training_complete(self, training_results):
        trained_model, train_losses, val_accuracies = training_results
        self.result_label.setText(f"Eğitim tamamlandı. Loss: {train_losses[-1]}, Accuracy: {val_accuracies[-1]}%")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Görüntüyü Seç", "", "Image files (*.jpg *.jpeg *.png)")
        if file_path:
            self.display_image(file_path)
            self.test_image(file_path)

    def test_image(self, file_path):
        self.result_label.setStyleSheet("QLabel { color: black; }")
        self.result_label.setText(f"Test Ediliyor...")
        # TestThread oluştur ve başlat
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.test_thread = TestThread(file_path, "vit_chest_xray_model.pth")
        self.test_thread.result_ready.connect(self.show_result)  # Sonucu göstermek için sinyali bağla.
        self.test_thread.start()

    def show_result(self, result):
        if result == "NORMAL":
            self.result_label.setStyleSheet("QLabel { color: green; }")
        else:
            self.result_label.setStyleSheet("QLabel { color: red; }")
        self.result_label.setText(f"Test sonucu: {result}")


    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(400, 400, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)  # Resmi yeniden boyutlandır ve yumuşat
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)



class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChestXRayApp()
    ex.show()
    sys.exit(app.exec_())
