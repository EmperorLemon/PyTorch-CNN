import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai.model import MLP, VGG16, PretrainedVGG16
from ai.train import Trainer
from ai.data import ImageDataset
from ai.utils import check_cuda, load_state
from ai.test import evaluate_model
from ai.visualizer import visualize_results

from torchinfo import summary

from tensorboardX import SummaryWriter

from utils import get_log_dir
from config import *
from globals import *

def main() -> int:
    check_cuda()

    # Load the dataset
    dataset = ImageDataset(FASHION_DATA_DIR, 
                           val_split=hyperparameters.get("val_test_split")[0], 
                           test_split=hyperparameters.get("val_test_split")[1])
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=hyperparameters.get("batch_size"), num_workers=4)

    num_classes = len(dataset.class_names)

    log_dir = get_log_dir()

    model = None

    # Create the model
    match hyperparameters.get("model_type"):
        case ModelType.MLP:
            model = MLP(3 * 224 * 224, 
                        hidden_layers=hyperparameters.get("hidden_layers"), 
                        num_classes=num_classes, 
                        dropout_rate=hyperparameters.get("dropout_rate"))
        case ModelType.VGG16:
            model = VGG16(num_classes=num_classes,
                          dropout_rate=hyperparameters.get("dropout_rate"))
        case ModelType.VGG16_TORCH:
            model = PretrainedVGG16(num_classes=num_classes)

    summary(model, input_size=(hyperparameters.get("batch_size"), 3, 224, 224))
    writer = SummaryWriter(log_dir=log_dir)

    if CURRENT_MODEL_MODE is ModelMode.TRAINING:
        # Create the trainer
        trainer = Trainer(n_epochs=hyperparameters.get("num_epochs"), 
                        lr=hyperparameters.get("learning_rate"), 
                        weight_decay=hyperparameters.get("weight_decay"),
                        device=model.device, writer=writer)

        # Train the model to fit the parameters
        trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader, optim_type=hyperparameters.get("optim_type"), pretrained=False)
    
        # Evaluate the effectiveness of the model
        conf_matrix, mae = evaluate_model(model=model, test_loader=test_loader)
        visualize_results(model, test_loader, conf_matrix, mae, writer=writer)
    elif CURRENT_MODEL_MODE is ModelMode.INFERENCE:
        model_state = None
        
        match hyperparameters.get("model_type"):
            case ModelType.MLP:
                model_state = load_state("mlp_best.pth", checkpoint=False)
            case ModelType.VGG16:
                model_state = load_state("vgg16_best.pth", checkpoint=False)
            case ModelType.VGG16_TORCH:
                model_state = load_state("vgg16_torch_best.pth", checkpoint=False)
                
        model.load_state_dict(model_state["model_state_dict"])
        
        epoch = model_state["epoch"]
        train_loss = model_state["train_loss"]
        val_loss = model_state["val_loss"]
        val_accuracy = model_state["val_accuracy"]
    
        print(f"Loaded model from epoch {epoch} with train loss {train_loss:.4f}, validation loss {val_loss:.4f}, and accuracy {val_accuracy:.2f}%")
        
        # Evaluate the effectiveness of the model
        conf_matrix, mae = evaluate_model(model=model, test_loader=test_loader)
        visualize_results(model, test_loader, conf_matrix, mae, writer=writer)

    writer.flush()
    writer.close()

    return 0

if __name__ == "__main__":
    main()