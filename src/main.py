from ai.model import Model, Layer, nn
from ai.train import Trainer
from ai.data import ImageDataset
from ai.utils import check_cuda, load_model, save_model
from ai.visualizer import visualize_results
from ai.test import verify_data_and_model, evaluate_model

from torch.utils.tensorboard.writer import SummaryWriter

from typing import List

from utils import prompt_user, get_log_dir
from config import hyperparameters
from globals import *

def train_model(model: Model, trainer: Trainer, train_loader, val_loader):
    trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader)

    save_model(model=model)

def test_model(model: Model, dataset, test_loader):
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name} device: {param.device}")

    verify_data_and_model(dataset=dataset, test_loader=test_loader, model=model)
    evaluate_model(dataset=dataset, test_loader=test_loader, model=model)

    # Visualize some results
    visualize_results(dataset=dataset, test_loader=test_loader, model=model, output_dir=OUTPUT_DIR, num_samples=5)

def main() -> int:
    check_cuda()

    # Load the dataset
    dataset = ImageDataset(FASHION_DATA_DIR, img_size=224, use_augmentation=True)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=hyperparameters.get("batch_size"), num_workers=4)

    input_size = 224 * 224 * 3
    output_size = len(dataset.class_names)

    network_layers : list[Layer] = [
        Layer("flatten", nn.Flatten()),
        Layer("fc1", nn.Linear(input_size, 1024)),
        Layer("relu1", nn.ReLU()),
        Layer("drop1", nn.Dropout(0.5)),
        Layer("fc2", nn.Linear(1024, 512)),
        Layer("relu2", nn.ReLU()),
        Layer("drop2", nn.Dropout(0.5)),
        Layer("fc3", nn.Linear(512, 256)),
        Layer("relu3", nn.ReLU()),
        Layer("drop3", nn.Dropout(0.5)),
        Layer("fc4", nn.Linear(256, output_size))
    ]

    log_dir = get_log_dir()
    writer = SummaryWriter(log_dir=log_dir)

    # Create the model
    model = Model(network_layers)
    print(f"\nModel created with device: {model.device}\n")

    # Create the trainer
    trainer = Trainer(n_epochs=hyperparameters.get("num_epochs"), 
                      lr=hyperparameters.get("learning_rate"), 
                      weight_decay=hyperparameters.get("weight_decay"), device=model.device, writer=writer)

    train_model(model=model, trainer=trainer, train_loader=train_loader, val_loader=val_loader)
    # model = load_model(model)

    # match prompt_user(model_path=MODEL_PATH):
    #     case 0:
    #         # Train the model
    #     case 1:
    #         model = load_model(model)
    #     case _:
    #         pass

    # Test the model
    test_model(model=model, dataset=dataset, test_loader=test_loader)

    writer.close()

    return 0

if __name__ == "__main__":
    main()