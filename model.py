import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


torch.cuda.manual_seed(5)
torch.cuda.manual_seed_all(5)  # if you are using multi-GPU.
np.random.seed(5)  # Numpy module.
# random.seed(10)  # Python random module.
torch.manual_seed(5)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#For single layer hidden model comment lines 43 and 44 in model.py, line 46 in main_combined.py, line 43 in main.py.

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        # Telling what are the specifications of the model
        #self.fc1 = nn.Linear(input, output)

        #1 Hidden Layer Model
        self.embedding = nn.Linear(args.num_param, args.out_lay1) #layer1
        self.output = nn.Linear(args.out_lay1, args.output_dim, bias=False)

        #2 Hidden Layer Model
        # self.embedding = nn.Linear(args.num_param, args.out_lay1)
        # self.layer3 = nn.Linear(args.out_lay1, args.out_lay3)
        # self.output = nn.Linear(args.out_lay3, args.output_dim, bias=False)

        # # #old Hidden Layer Model
        # self.embedding = nn.Linear(args.num_param, args.out_lay1) #layer1
        # # # self.layer2 = nn.Linear(args.out_lay1, args.out_lay2)
        # # # self.layer3 = nn.Linear(args.out_lay2, args.out_lay3)
        # self.layer3 = nn.Linear(args.out_lay1, args.out_lay3) #2 lay
        # self.output = nn.Linear(args.out_lay3, args.output_dim, bias=False) #2 lay
        # # self.output = nn.Linear(args.out_lay1, args.output_dim, bias=False)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(args.dropout)

    #Telling model what to do with the layers
    def forward(self, x, hid_out=False):

        y = self.embedding(x)
        y = F.relu(y)
        y = self.dropout(y)
        # y = F.relu(self.layer2(y)) #comment this line for 1 Hidden Layer Model
        # y = self.dropout(y) #comment this line for 1 Hidden Layer Model
        # y = F.relu(self.layer3(y))
        # y = self.dropout(y)
        if (hid_out == True):
            return y
        y = self.output(y)

        return y


class Net_combined(nn.Module):

    def __init__(self, args, model1, model2, model3, model4, model5, model6):
        super(Net_combined, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6

        # Freeze output from out_lay3
        for n,p in self.model1.named_parameters():
            p.requires_grad = False
        for n,p in self.model2.named_parameters():
            p.requires_grad = False
        for n,p in self.model3.named_parameters():
            p.requires_grad = False
        for n,p in self.model4.named_parameters():
            p.requires_grad = False
        for n,p in self.model5.named_parameters():
            p.requires_grad = False
        for n,p in self.model6.named_parameters():
            p.requires_grad = False #comment until here to try
        self.out_lay3 = args.out_lay3 #2 Layer
        # self.out_lay1 = args.out_lay1 #1 Layer

        #2 Layer
        self.output = nn.Linear(self.out_lay3*6, args.output_dim)
        #Attention Layer
        self.attention_weight = nn.Linear(args.out_lay3, 6) # bs x 6

        # #1 Layer
        # self.output = nn.Linear(self.out_lay1*6, args.output_dim)
        # #Attention Layer
        # self.attention_weight = nn.Linear(args.out_lay1, 6) # 256 x 6

    #Telling model what to do with the layers
    def forward(self, x1,x2,x3,x4,x5,x6): #model_RPPA, model_Meta, model_Mut, model_Exp, model_CNV,model_miRNA
        y1 = self.model1(x1, hid_out = True)
        y2 = self.model2(x2, hid_out = True)
        y3 = self.model3(x3, hid_out = True)
        y4 = self.model4(x4, hid_out = True)
        y5 = self.model5(x5, hid_out = True)
        y6 = self.model6(x6, hid_out = True)
        #Concatenate
        y=torch.cat([y1,y2,y3,y4,y5,y6],dim=-1) # bs x 384
        # print(y.shape)

        #Attention Mechanism
        y = y.reshape((y.shape[0], 6,self.out_lay3)) # bs x 6 x 64
        # print(y.shape)
        attention_weight = self.attention_weight(y) # bs x 6 , Get attention weight
        # print(attention_weight.shape)
        softmax = F.softmax(attention_weight,dim=-1) # bs x 6, Convert to probability (0-1) through softmax
        # print(softmax.shape)
        y = torch.bmm(softmax, y) #batch matrix mulitplication of the 6 output (models) and attention weight
        # print(y.shape)
        y = y.reshape(-1, self.out_lay3 *6) # bs x 384, Aft multiplication by attention weight, rehsape to orginal shape
        # print(y.shape) # comment until here to check without attention

        # #1 Layer
        # #Attention Mechanism
        # y = y.reshape((y.shape[0], 6,self.out_lay1)) # bs x 6 x 64
        # # print(y.shape)
        # attention_weight = self.attention_weight(y) # bs x 6 , Get attention weight
        # # print(attention_weight.shape)
        # softmax = F.softmax(attention_weight,dim=-1) # bs x 6, Convert to probability (0-1) through softmax
        # # print(softmax.shape)
        # y = torch.bmm(softmax, y) #batch matrix mulitplication of the 6 output (models) and attention weight
        # # print(y.shape)
        # y = y.reshape(-1, self.out_lay1 *6) # bs x 384, Aft multiplication by attention weight, rehsape to orginal shape
        # # print(y.shape) # comment until here to check without attention

        output = self.output(y)
        # print(output.shape)

        return output

class Net_CNN(nn.Module):

    def __init__(self, args):
        super(Net_CNN, self).__init__()

        # 2 Hidden Layer Model
        self.embedding = nn.Linear(args.num_param, args.out_lay1)
        self.layer2 = nn.Conv1d(in_channels=args.out_lay1, out_channels=args.out_lay2,kernel_size =1)
        self.layer3 = nn.Conv1d(in_channels=args.out_lay2, out_channels=args.out_lay3, kernel_size =1)
        self.output = nn.Linear(args.out_lay3, args.output_dim)

        # #1 Hidden Layer Model
        # self.embedding = nn.Linear(args.num_param, args.out_lay1)
        # self.layer3 = nn.Conv1d(in_channels=args.out_lay1, out_channels=args.out_lay3, kernel_size =1)
        # self.output = nn.Linear(args.out_lay3, args.output_dim)


        # Define proportion of neurons to dropout
        self.dropout = nn.Dropout(args.dropout)

    #Telling model what to do with the layers
    def forward(self, x, hid_out=False):

        y = self.embedding(x).unsqueeze(-1) #bs x 200 x 1
        y = F.relu(y)
        y = self.dropout(y)
        y = F.relu(self.layer2(y)) #comment this line for 1 Hidden Layer Model
        y = self.dropout(y) #comment this line for 1 Hidden Layer Model
        y = F.relu(self.layer3(y)).squeeze()
        y = self.dropout(y)
        if (hid_out == True):
            return y
        y = self.output(y)

        return y
