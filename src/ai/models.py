from .model import Layer, nn
from typing import List

def make_lazy_vgg_block(out_channels: int, num_convs: int, block_num: int) -> List[Layer]:
    layers = []

    for i in range(num_convs):
        layers.append(Layer(f"conv{block_num}_{i+1}", nn.LazyConv2d(out_channels, kernel_size=3, padding=1)))
        layers.append(Layer(f"batch{block_num}_{i+1}", nn.LazyBatchNorm2d()))
        layers.append(Layer(f"relu{block_num}_{i+1}", nn.ReLU()))

    layers.append(Layer(f"pool{block_num}", nn.MaxPool2d(kernel_size=2, stride=2)))

    return layers

def create_lazy_vgg16(output_size: int) -> List[Layer]:
    # VGG16 architecture with lazy layers
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