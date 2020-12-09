import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        # Telling what are the specifications of the model
        # num_parameter,out_embedding=200,out_layer2=128,out_layer3=32,output_dim=22
        self.embedding = nn.Linear(args.num_param, args.out_embed)
        self.layer2 = nn.Linear(args.out_embed, args.out_lay2)
        self.layer3 = nn.Linear(args.out_lay2, args.out_lay3)
        self.output = nn.Linear(args.out_lay3, args.output_dim)
    #Telling model what to do with the layers
    def forward(self, x):
        y = self.embedding(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.output(y)

        return y
