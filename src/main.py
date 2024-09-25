import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai.model import Model, VGG16, PretrainedVGG16
from ai.train import Trainer
from ai.data import ImageDataset
from ai.utils import check_cuda, load_state
from ai.test import evaluate_model
from ai.models import create_mlp

from torchinfo import summary

from tensorboardX import SummaryWriter

from typing import List
from utils import get_log_dir
from config import hyperparameters
from globals import *

def train_model(model, trainer: Trainer, train_loader, val_loader):
    trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader)

def test_model(test_loader, model):
    evaluate_model(test_loader=test_loader, model=model)

def main() -> int:
    check_cuda()

    # Load the dataset
    dataset = ImageDataset(FASHION_DATA_DIR)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=hyperparameters.get("batch_size"), num_workers=4)

    fc_layers: List[int] = [512, 256]
    output_size = len(dataset.class_names)
    
    network_layers = create_mlp(3 * 224 * 224, fc_layers=fc_layers, num_classes=output_size)

    log_dir = get_log_dir()
    writer = SummaryWriter(log_dir=log_dir)

    # Create the model
    model = Model(layers=network_layers)
    # model = VGG16(num_classes=output_size)
    # model = PretrainedVGG16(num_classes=output_size)

    # summary(model, input_size=(hyperparameters.get("batch_size"), IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT))

    # model_state = load_state("vgg_best.pth", True)
    # model.load_state_dict(model_state["model_state_dict"])

    # Create the trainer
    trainer = Trainer(n_epochs=hyperparameters.get("num_epochs"), 
                      lr=hyperparameters.get("learning_rate"), 
                      weight_decay=hyperparameters.get("weight_decay"), 
                      device=model.device, writer=writer)

    train_model(model=model, trainer=trainer, train_loader=train_loader, val_loader=val_loader)

    test_model(test_loader=test_loader, model=model)

    writer.flush()
    writer.close()

    return 0

if __name__ == "__main__":
    main()