import torch
from torch import nn

class new_class(nn.Module):
    def __init__(self):
        super(new_class, self).__init__()

    def forward(self, input):
        return input+1

if __name__=='__main__':
    new_model = new_class()
    x = torch.tensor(1.0)
    output = new_model(x)
    print(output)