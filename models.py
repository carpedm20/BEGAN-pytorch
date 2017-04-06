import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class BaseModel(nn.Module):
    def forward(self, x):
        gpu_ids = None
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            gpu_ids = range(self.num_gpu)
        if gpu_ids:
            return nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            return self.main(x)

class GeneratorCNN(BaseModel):
    def __init__(self, input_num, initial_conv_dim, output_num, repeat_num, hidden_num, num_gpu):
        super(GeneratorCNN, self).__init__()
        self.num_gpu = num_gpu
        layers = []

        self.initial_conv_dim = initial_conv_dim
        self.fc = nn.Linear(input_num, np.prod(self.initial_conv_dim))
        
        layers = []
        for idx in range(repeat_num):
            layers.append(nn.Conv2d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))

            if idx < repeat_num - 1:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        layers.append(nn.Conv2d(hidden_num, output_num, 3, 1, 1))
        #layers.append(nn.Tanh())
        layers.append(nn.ELU(True))

        self.conv = torch.nn.Sequential(*layers)
        
    def main(self, x):
        fc_out = self.fc(x).view([-1] + self.initial_conv_dim)
        return self.conv(fc_out)

class DiscriminatorCNN(BaseModel):
    def __init__(self, input_channel, z_num, repeat_num, hidden_num, num_gpu):
        super(DiscriminatorCNN, self).__init__()
        self.num_gpu = num_gpu

        # Encoder
        layers = []
        layers.append(nn.Conv2d(input_channel, hidden_num, 3, 1, 1))
        layers.append(nn.ELU(True))

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            layers.append(nn.Conv2d(prev_channel_num, channel_num, 3, 1, 1))
            layers.append(nn.ELU(True))

            if idx < repeat_num - 1:
                layers.append(nn.Conv2d(channel_num, channel_num, 3, 2, 1))
                #layers.append(nn.MaxPool2d(2))
                #layers.append(nn.MaxPool2d(1, 2))
            else:
                layers.append(nn.Conv2d(channel_num, channel_num, 3, 1, 1))

            layers.append(nn.ELU(True))
            prev_channel_num = channel_num

        self.conv1_output_dim = [channel_num, 8, 8]

        self.conv1 = torch.nn.Sequential(*layers)
        self.fc1 = nn.Linear(8*8*channel_num, z_num)

        # Decoder
        self.conv2_input_dim = [hidden_num, 8, 8]
        self.fc2 = nn.Linear(z_num, np.prod(self.conv2_input_dim))
        
        layers = []
        for idx in range(repeat_num):
            layers.append(nn.Conv2d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))

            if idx < repeat_num - 1:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        layers.append(nn.Conv2d(hidden_num, input_channel, 3, 1, 1))
        #layers.append(nn.Tanh())
        layers.append(nn.ELU(True))

        self.conv2 = torch.nn.Sequential(*layers)

    def main(self, x):
        conv1_out = self.conv1(x).view(-1, np.prod(self.conv1_output_dim))
        fc1_out = self.fc1(conv1_out)

        fc2_out = self.fc2(fc1_out).view([-1] + self.conv2_input_dim)
        conv2_out = self.conv2(fc2_out)
        return conv2_out

class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # this won't still solve the problem
        # which means gradient will not flow through target
        # _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument `sizeAverage=False`
    """
    pass
