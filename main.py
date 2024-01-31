from DataLoader import DataLoader
from ModelTrainer import ChestXRayModelTrainer

train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

def prepare_data_loaders(train_dir, val_dir, test_dir):
    """Veri yükleyicileri hazırla ve döndür."""
    data_loader = DataLoader(train_dir, val_dir, test_dir)
    return data_loader.create_dataloaders(batch_size=32)

def train_and_evaluate_model(train_loader, val_loader, test_loader):
    """Modeli eğit, değerlendir ve döndür."""
    model_trainer = ChestXRayModelTrainer(train_loader, val_loader)
    trained_model, train_losses, val_accuracies = model_trainer.train(num_epochs=1)
    test_accuracy = model_trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    trained_model.save_pretrained('ModelFineTuning')
    return trained_model, train_losses, val_accuracies, test_accuracy

def train_model(train_dir, val_dir, test_dir):
    train_loader, val_loader, test_loader = prepare_data_loaders(train_dir, val_dir, test_dir)
    trained_model, train_losses, val_accuracies, \
        test_accuracy = train_and_evaluate_model(train_loader, val_loader, test_loader)

def main():

    train_model(train_dir, val_dir, test_dir)
    return 0

if __name__ == "__main__":
    main()