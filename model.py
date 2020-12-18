import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(10)
np.random.seed(10)

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        # Telling what are the specifications of the model
        # num_parameter,out_embedding=200,out_layer2=128,out_layer3=32,output_dim=22
        #self.fc1 = nn.Linear(input, output)
        self.embedding = nn.Linear(args.num_param, args.out_embed)
        self.layer2 = nn.Linear(args.out_embed, args.out_lay2)
        self.layer3 = nn.Linear(args.out_lay2, args.out_lay3)
        self.output = nn.Linear(args.out_lay3, args.output_dim)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

    #Telling model what to do with the layers
    def forward(self, x, hid_out=False):
        #activation function?
        y = self.embedding(x)
        y = F.relu(self.layer2(y))
        y = self.dropout(y)
        y = F.relu(self.layer3(y))
        y = self.dropout(y)
        if (hid_out == True):
            return y
        y = self.output(y)

        return y

class Net_combined(nn.Module):

    def __init__(self, args, model1, model2, model3, model4, model5):
        super(Net_combined, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.output = nn.Linear(5, args.output_dim)

        self.attention_weight = nn.Parameter(torch.FloatTensor( args.out_lay3*5,5))

    #Telling model what to do with the layers
    def forward(self, x1,x2,x3,x4,x5): #model_RPPA, model_Meta, model_Mut, model_Exp, model_CNV
        y1 = self.model1(x1, hid_out = True)
        y2 = self.model2(x2, hid_out = True)
        y3 = self.model3(x3, hid_out = True)
        y4 = self.model4(x4, hid_out = True)
        y5 = self.model5(x5, hid_out = True)
        #Concatenate
        y=torch.cat([y1,y2,y3,y4,y5],dim=-1) # bs x 320

        attention_weight = torch.matmul(y, self.attention_weight) #bs x 5
        softmax = F.softmax(attention_weight,dim=-1)
        output = self.output(softmax)

        return output



        return y

class Net_CNN(nn.Module):

    def __init__(self, args):
        super(Net_CNN, self).__init__()
        # Telling what are the specifications of the model
        # num_parameter,out_embedding=200,out_layer2=128,out_layer3=32,output_dim=22
        #self.fc1 = nn.Linear(input, output)
        self.embedding = nn.Linear(args.num_param, args.out_embed)
        self.layer2 = nn.Conv1d(in_channels=args.out_embed, out_channels=args.out_lay2,kernel_size =1)
        self.layer3 = nn.Conv1d(in_channels=args.out_lay2, out_channels=args.out_lay3, kernel_size =1)
        self.output = nn.Linear(args.out_lay3, args.output_dim)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

    #Telling model what to do with the layers
    def forward(self, x, hid_out=False):
        #activation function?
        y = self.embedding(x).unsqueeze(-1) #bs x 200 x 1
        y = F.relu(self.layer2(y)) #
        y = self.dropout(y)
        y = F.relu(self.layer3(y)).squeeze()
        y = self.dropout(y)
        if (hid_out == True):
            return y
        y = self.output(y)

        return y
