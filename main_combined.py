def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from model import Net, Net_combined
import os
from datasets import *
import torch.optim
import torch.nn as nn
import numpy as np
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
torch.manual_seed(10)
np.random.seed(10)


parser = argparse.ArgumentParser(description='Drug Response Prediction')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--esthres', type=int, default=5, help='')
# parser.add_argument('--est', type=int, default=30, help='early_stopping_threshold')

parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')
parser.add_argument('--data', type=str, default='RPPA', help='')
parser.add_argument('--expr_dir', type=str, default="experiments/", help='path/to/save_dir')
parser.add_argument('--cv', action='store_true', help='cross validation') #--cv
# parser.add_argument('--num_param', type=int, default =101) #Remember to change parameter when reading diff dataset

parser.add_argument('--out_embed', type=int, default=200)
parser.add_argument('--out_lay2', type=int, default =128)
parser.add_argument('--out_lay3', type=int, default =64)
parser.add_argument('--output_dim',type=int, default=24)

# num_parameter,out_embedding=200,out_layer2=128,out_layer3=32,output_dim=22

def main(args, train_data,valid_data,test_data):

    ### init training and val stuff ###
    #Loss Function
    criterion = nn.MSELoss()
    train_data,valid_data,test_data = read_combined()

    args.num_param = 101
    model_RPPA = Net(args)
    model_RPPA = load_checkpoint('experiments/RPPA/model_best.pth.tar', model_RPPA)


    args.num_param = 80
    model_Meta = Net(args)
    model_Meta = load_checkpoint('experiments/Meta/model_best.pth.tar', model_Meta)


    args.num_param = 1040
    model_Mut = Net(args)
    model_Mut = load_checkpoint('experiments/Mut/model_best.pth.tar', model_Mut)


    args.num_param = 88
    model_CNV = Net(args)
    model_CNV = load_checkpoint('experiments/CNV/model_best.pth.tar', model_CNV)

    args.num_param = 616
    model_Exp = Net(args)
    model_Exp = load_checkpoint('experiments/Exp/model_best.pth.tar', model_Exp)

    args.is_cuda = torch.cuda.is_available()

    model = Net_combined(args, model_RPPA, model_Meta, model_Mut, model_Exp, model_CNV)
    if args.is_cuda:
        model = model.cuda()

    #Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_loader = torch.utils.data.DataLoader(TensorDataset(train_data[0],train_data[1], train_data[2],train_data[3],train_data[4],train_data[5]), batch_size=args.batchSize, shuffle = True,num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(TensorDataset(valid_data[0],valid_data[1], valid_data[2],valid_data[3],valid_data[4],valid_data[5]), batch_size=args.batchSize, shuffle = True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_data[0],test_data[1], test_data[2],test_data[3],test_data[4],test_data[5]), batch_size=args.batchSize, shuffle = True, num_workers=0, pin_memory=True)

    ### print options ###

    all_train_loss, all_valid_loss, all_test_loss = [],[],[]
    all_train_TT = []
    best_valid_loss = 99999
    test_loss_best_val = 99999
    count = 0

    for epoch in range(0,args.epoch):

        # train for one epoch
        # returns total_loss and TT for 1 epoch
        epoch_total_loss, TT = train(train_loader, model, criterion, optimizer, epoch, args)
        #loss for total epochs
        all_train_loss.append(epoch_total_loss)
        #TT for total epochs
        all_train_TT.append(TT)

        # evaluate on validation set
        epoch_val_loss = validate(val_loader, model, criterion, args)
        all_valid_loss.append(epoch_val_loss)

        # evaluate on test set
        epoch_test_loss = validate(test_loader, model, criterion, args)
        all_test_loss.append(epoch_test_loss)

        #Early Stopping
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }
        if(epoch_val_loss < best_valid_loss):
            count = 0
            best_valid_loss = epoch_val_loss
            test_loss_best_val = epoch_test_loss

            save_checkpoint(state, True, args)
        else:
            count = count + 1



            if(count >= args.esthres):
                break

    #plotting code
    #plotting the training and validation loss
    plt.plot(all_train_loss, label='Training loss')
    plt.plot(all_valid_loss, label='Validation loss')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('No. of epochs')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(os.path.join(args.expr_dir, 'testplt1.png'))


    return best_valid_loss, test_loss_best_val

def train(train_loader, model, criterion, optimizer, epoch, args):
    total_loss = 0.0
    # switch to train mode
    model.train()
    stime = time.time()
    #Updating the parameters each iteration. (# of iterations = # batches)
    #Each iteration: 1)Forward Propogation 2)Compute Costs 3)Backpropagation 4)Update parameters
    for i, (input1,input2,input3,input4,input5, target) in enumerate(train_loader):
        # measure data loading time
        #???
        target = target.float()
        input1,input2,input3,input4,input5 = input1.float(),input2.float(),input3.float(),input4.float(),input5.float()
        if args.is_cuda:
            target = target.cuda()
            input1,input2,input3,input4,input5 = input1.cuda(),input2.cuda(),input3.cuda(),input4.cuda(),input5.cuda()

        #Forward pass to compute output
        output = model(input1,input2,input3,input4,input5)
        #Calculate Loss: MSE
        loss = criterion(output, target)
        #Adding loss for current iteration into total_loss
        total_loss += loss
        #Clear gradients w.r.t parameters
        optimizer.zero_grad()
        #Getting gradients w.r.t parameters
        loss.backward()
        #Updating parameters
        optimizer.step()
    #Time taken for 1 epoch
    TT = time.time() -stime
    #Avg. total_loss for all the iteration/loss for 1 epoch
    total_loss =  total_loss/(i+1)

    print('Epoch: [{0}]\t'
          'Training Loss {loss:.3f}\t'
          'Time: {time:.2f}\t'.format(
           epoch, loss=total_loss, time= TT))

    return total_loss, TT




def validate(val_loader, model, criterion, args):

    # switch to evaluate mode
    model.eval()

    total_loss = 0.0
    #Prevent tracking history,for validation.
    with torch.no_grad():

        for i, (input1,input2,input3,input4,input5, target) in enumerate(val_loader):

            target = target.float()
            input1,input2,input3,input4,input5 = input1.float(),input2.float(),input3.float(),input4.float(),input5.float()
            if args.is_cuda:
                target = target.cuda()
                input1,input2,input3,input4,input5 = input1.cuda(),input2.cuda(),input3.cuda(),input4.cuda(),input5.cuda()

            # Forward pass to compute output
            output = model(input1,input2,input3,input4,input5)
            #Calculate Loss: MSE
            loss = criterion(output, target)
            total_loss += loss


    total_loss =  total_loss/(i+1)
    print('val: \t'
          'Loss {loss:.4f}\t'.format(loss=total_loss))

    return total_loss

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.expr_dir, filename))
    if is_best:
        torch.save(state, os.path.join(args.expr_dir, 'model_best.pth.tar'))
def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    #1) data loading
    if args.cv:
        #train_val_data,test_data - returns 5 features and 1 labe;
        train_val_data, train_val_label, test_data, test_label = read_combined(cv=args.cv)
    else:
        train_data,valid_data,test_data = read_combined(cv=args.cv)

    if args.cv:
        #2) Perform kfold cross-validation
        kf = KFold(n_splits=5)
        kf.get_n_splits(train_val_data)

        best_valid_results = []
        test_loss_best_val_results = []

        test_data_RPPA, test_data_Meta, test_data_Mut, test_data_Exp, test_data_CNV = test_data[:,:101],test_data[:,101:181],test_data[:,181:1221],test_data[:,1221:1837],test_data[:,1837:1925]
        test_data = (torch.tensor(test_data_RPPA), torch.tensor(test_data_Meta), torch.tensor(test_data_Mut), torch.tensor(test_data_Exp), torch.tensor(test_data_CNV),torch.tensor(test_label))
        #In this case text_index is my val_index
        for train_index, test_index in kf.split(train_val_data):

            # print(train_index)
            # print(test_index)

            train_data, val_data = train_val_data[train_index], train_val_data[test_index]
            train_label, val_label = train_val_label[train_index], train_val_label[test_index]

            #Differentiating features and labels
            train_data_RPPA, train_data_Meta, train_data_Mut, train_data_Exp, train_data_CNV = train_data[:,:101],train_data[:,101:181],train_data[:,181:1221],train_data[:,1221:1837],train_data[:,1837:1925]
            valid_data_RPPA, valid_data_Meta, valid_data_Mut, valid_data_Exp, valid_data_CNV = val_data[:,:101],val_data[:,101:181],val_data[:,181:1221],val_data[:,1221:1837],val_data[:,1837:1925]

        
            train_data = (torch.tensor(train_data_RPPA), torch.tensor(train_data_Meta), torch.tensor(train_data_Mut), torch.tensor(train_data_Exp), torch.tensor(train_data_CNV),torch.tensor(train_label))
            valid_data = (torch.tensor(valid_data_RPPA), torch.tensor(valid_data_Meta), torch.tensor(valid_data_Mut), torch.tensor(valid_data_Exp), torch.tensor(valid_data_CNV),torch.tensor(val_label))


            best_valid_loss, test_loss_best_val = main(args, train_data,valid_data,test_data)

            best_valid_results.append(best_valid_loss)
            test_loss_best_val_results.append(test_loss_best_val)

        #Get avg. scores obtained across the k-folds
        best_valid_average = np.mean(best_valid_results)
        test_loss_best_val_average = np.mean(test_loss_best_val_results)

        print("best_valid_loss_CV_avg:", best_valid_average,  "test_loss_best_val_CV_avg:", test_loss_best_val_average)
    else:
        best_valid_loss, test_loss_best_val = main(args, train_data,valid_data,test_data)
        print("best_valid_loss:", best_valid_loss,  "test_loss_best_val:", test_loss_best_val)

    # best_valid_loss, test_loss_best_val = main(args)
    # print("best_valid_loss:", best_valid_loss,  "test_loss_best_val:", test_loss_best_val)
