from ai.model import Model
from ai.train import Trainer
from ai.data import ImageDataset
from ai.utils import check_cuda, save_state, load_state
from ai.visualizer import visualize_results
from ai.test import verify_data_and_model, evaluate_model
from ai.models import create_lazy_vgg16, create_lazy_mlp

from torch.utils.tensorboard.writer import SummaryWriter

from typing import List

from utils import prompt_user, get_log_dir
from config import hyperparameters
from globals import *

def train_model(model: Model, trainer: Trainer, train_loader, val_loader):
    trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader)

    # save_state(state=model.state_dict(), model_file="<file>.pth")

def test_model(model: Model, dataset, test_loader):
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name} device: {param.device}")

    verify_data_and_model(dataset=dataset, test_loader=test_loader, model=model)
    evaluate_model(dataset=dataset, test_loader=test_loader, model=model)

    # Visualize some results
    visualize_results(dataset=dataset, test_loader=test_loader, model=model, output_dir=OUTPUT_DIR, num_samples=10)

def main() -> int:
    check_cuda()

    # Load the dataset
    dataset = ImageDataset(FASHION_DATA_DIR, img_size=32, use_augmentation=True)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=hyperparameters.get("batch_size"), num_workers=4)

    output_size = len(dataset.class_names)
    hidden_layers = [64, 32]

    # VGG16 architecture
    # network_layers = create_lazy_vgg16(output_size=output_size)
    network_layers = create_lazy_mlp(hidden_layers=hidden_layers, output_size=output_size, dropout_rate=0.5)

    log_dir = get_log_dir()
    writer = SummaryWriter(log_dir=log_dir)

    # Create the model
    model = Model(network_layers)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")

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

    writer.flush()
    writer.close()

    return 0

if __name__ == "__main__":
    main()