from .model import Layer, nn
from typing import Type, List

def make_lazy_vgg_block(out_channels: int, num_convs: int, block_num: int) -> List[Layer]:
    layers: List[Layer] = []

    for i in range(num_convs):
        layers.append(Layer(f"conv{block_num}_{i+1}", nn.LazyConv2d(out_channels, kernel_size=3, padding=1)))
        layers.append(Layer(f"batch{block_num}_{i+1}", nn.LazyBatchNorm2d()))
        layers.append(Layer(f"relu{block_num}_{i+1}", nn.ReLU()))

    layers.append(Layer(f"pool{block_num}", nn.MaxPool2d(kernel_size=2, stride=2)))

    return layers

## VGG16 architecture with lazy layers
def create_lazy_vgg16(output_size: int) -> List[Layer]:
    network_layers: List[Layer] = (
        make_lazy_vgg_block(64, 2, 1) +
        make_lazy_vgg_block(128, 2, 2) +
        make_lazy_vgg_block(256, 3, 3) +
        make_lazy_vgg_block(512, 3, 4) +
        make_lazy_vgg_block(512, 3, 5) +
        [
            Layer("flatten", nn.Flatten()),
            Layer("fc1", nn.LazyLinear(4096)),
            Layer("relu_fc1", nn.ReLU()),
            Layer("drop1", nn.Dropout(0.5)),
            Layer("fc2", nn.LazyLinear(4096)),
            Layer("relu_fc2", nn.ReLU()),
            Layer("drop2", nn.Dropout(0.5)),
            Layer("fc3", nn.LazyLinear(output_size))
        ]
    )
    
    return network_layers

## Multilayer Perceptron architecture with lazy layers
def create_lazy_mlp(hidden_layers: List[int],
                    output_size: int,
                    dropout_rate: float = 0.5,
                    activation: Type[nn.Module] = nn.ReLU()) -> List[Layer]:
    network_layers: List[Layer] = []

    # Flatten Layer
    network_layers.append(Layer("flatten", nn.Flatten()))

    # Input layer
    network_layers.append(Layer("input", nn.LazyLinear(hidden_layers[0])))
    network_layers.append(Layer("act_input", activation))
    network_layers.append(Layer("drop_input", nn.Dropout(dropout_rate)))

    # Hidden layers
    for i, hidden_size in enumerate(hidden_layers[1:], 1):
        network_layers.append(Layer(f"hidden_{i}", nn.LazyLinear(hidden_size)))
        network_layers.append(Layer(f"act_{i}", activation))
        network_layers.append(Layer(f"drop_{i}", nn.Dropout(dropout_rate)))

    # Output layer
    network_layers.append(Layer("output", nn.LazyLinear(output_size)))

    # print([(layer.name, layer.module) for layer in network_layers])

    return network_layers

def create_lazy_cnn(conv_layers: List[int],
                    fc_layers: List[int],
                    output_size: int,
                    dropout_rate: float = 0.5,
                    activation: Type[nn.Module] = nn.ReLU()) -> List[Layer]:
    network_layers: List[Layer] = []

    # Convolutional layers
    for i, num_filters in enumerate(conv_layers):
        if i == 0:
            network_layers.append(Layer(f"conv_{i}", nn.LazyConv2d(num_filters, kernel_size=3, padding=1)))
        else:
            network_layers.append(Layer(f"conv_{i}", nn.LazyConv2d(num_filters, kernel_size=3, padding=1)))

        network_layers.append(Layer(f"bn_conv_{i}", nn.LazyBatchNorm2d()))
        network_layers.append(Layer(f"act_conv_{i}", activation))
        
        # Add pooling every other layer
        if i % 2 == 1:
            network_layers.append(Layer(f"pool_{i}", nn.MaxPool2d(2)))

        network_layers.append(Layer(f"drop_conv_{i}", nn.Dropout2d(dropout_rate)))

    # Flatten layer
    network_layers.append(Layer("flatten", nn.Flatten()))

    # Fully connected layers
    for i, fc_size in enumerate(fc_layers):
        network_layers.append(Layer(f"fc_{i}", nn.LazyLinear(fc_size)))
        network_layers.append(Layer(f"bn_fc_{i}", nn.LazyBatchNorm1d()))
        network_layers.append(Layer(f"act_fc_{i}", activation))
        network_layers.append(Layer(f"drop_fc_{i}", nn.Dropout(dropout_rate)))

    # Output layer
    network_layers.append(Layer("output", nn.LazyLinear(output_size)))

    # print([f"({layer.name}, {layer.module})" for layer in network_layers])

    return network_layers