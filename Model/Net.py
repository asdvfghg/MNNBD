import torch.nn as nn
from torchsummary import summary
from Model.frequency import *
class Net(nn.Module):
    def __init__(self, n_filter=40, fs=20000):
        super(Net, self).__init__()
        self.cnn1 = nn.Conv1d(1, 1, n_filter, 1, 'same', bias=False)
        self.bn1 = nn.BatchNorm1d(1)
        self.fs = fs

    def forward(self, input):
        y1 = self.cnn1(input)
        y1 = self.bn1(y1)
        envelope, freq = get_envelope_frequency(y1, self.fs)
        return y1, envelope



if __name__ == '__main__':
    x = torch.rand((1, 1, 2000))
    model = Net()
    y = model(x)
    print(y.shape)
    summary(model, (1, 20000), device='cpu')