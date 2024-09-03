from ai.model import Model, Layer, nn
from ai.train import Trainer
from ai.data import ImageDataset
from ai.utils import check_cuda, load_model
from ai.visualizer import visualize_results
from ai.test import evaluate_model

from torch.utils.tensorboard.writer import SummaryWriter

from typing import List

from utils import prompt_user
from globals import *

def train_model(model: Model, trainer: Trainer, train_loader, val_loader):
    trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader)

def test_model(model: Model, test_loader, class_names: List[str]):
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name} device: {param.device}")

    evaluate_model(test_loader=test_loader, model=model)

    # Visualize some results
    visualize_results(test_loader, model, OUTPUT_DIR, class_names=class_names, num_samples=10)

def main() -> int:
    check_cuda()

    writer = SummaryWriter(LOG_DIR)

    # Load the dataset
    dataset = ImageDataset(FASHION_DATA_DIR, img_size=224, use_augmentation=True)
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

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

    # Create the model
    model = Model(network_layers)
    print(f"Model created with device: {model.device}")

    # Create the trainer
    trainer = Trainer(n_epochs=100, lr=1e-3, weight_decay=1e-4, device=model.device, writer=writer)

    match prompt_user(model_path=MODEL_PATH):
        case 0:
            # Train the model
            train_model(model=model, trainer=trainer, train_loader=train_loader, val_loader=val_loader)
        case 1:
            model = load_model(model)
        case _:
            pass

    # Test the model
    test_model(model=model, test_loader=test_loader, class_names=dataset.class_names)

    writer.close()

    return 0

if __name__ == "__main__":
    main()