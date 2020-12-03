import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_parameter,out_embedding=200,out_layer2=128,out_layer3=32,output_dim=22):
        super(Net, self).__init__()
        # Telling what are the specifications of the model
        self.embedding = nn.Linear(num_parameter,out_embedding)
        self.layer2 = nn.Linear(out_embedding, out_layer2)
        self.layer3 = nn.Linear(out_layer2, out_layer3)
        self.output = nn.Linear(out_layer3, output_dim)
#Telling model what to do with the layers
    def forward(self, x):
        y = self.embedding(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.output(y)

        return y

#Creating the actual model
net = Net(num_parameter=20, output_dim=100)
print(net)
