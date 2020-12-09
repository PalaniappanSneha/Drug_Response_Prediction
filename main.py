def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from model import Net

from datasets import *
import torch.optim
import torch.nn as nn
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
torch.manual_seed(10)


parser = argparse.ArgumentParser(description='Drug Response Prediction')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--esthres', type=int, default=5, help='')
# parser.add_argument('--est', type=int, default=30, help='early_stopping_threshold')

parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')
parser.add_argument('--save_dir', type=str, default="experiments/", help='path/to/save_dir')
parser.add_argument('--num_param', type=int, default =101) #Remember to change parameter when reading diff dataset
parser.add_argument('--out_embed', type=int, default=200)
parser.add_argument('--out_lay2', type=int, default =128)
parser.add_argument('--out_lay3', type=int, default =32)
parser.add_argument('--output_dim',type=int, default=24)



# num_parameter,out_embedding=200,out_layer2=128,out_layer3=32,output_dim=22



def main(args):

    ### init training and val stuff ###
    #Loss Function
    criterion = nn.MSELoss()
    train_data, train_label, valid_data, valid_label, test_data, test_label = read_RPPA()

    args.is_cuda = torch.cuda.is_available()

    model = Net(args)     #force model to float and cuda
    if args.is_cuda:
        model = model.cuda()

    #Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_loader = torch.utils.data.DataLoader(TensorDataset(torch.tensor(train_data),torch.tensor(train_label)), batch_size=args.batchSize, shuffle = True,num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(TensorDataset(torch.tensor(test_data), torch.tensor(test_label)), batch_size=args.batchSize, shuffle = True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(TensorDataset(torch.tensor(valid_data),torch.tensor(valid_label)), batch_size=args.batchSize, shuffle = True, num_workers=0, pin_memory=True)

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
        all_valid_loss.append(epoch_total_loss)

        # evaluate on test set
        epoch_test_loss = validate(test_loader, model, criterion, args)
        all_test_loss.append(epoch_total_loss)

        #Early Stopping
        if(epoch_val_loss < best_valid_loss):
            count = 0
            best_valid_loss = epoch_val_loss
            test_loss_best_val = epoch_test_loss
        else:
            count = count + 1



            if(count >= args.esthres):
                break

    #plotting code
    # plotting the training and validation loss
    plt.plot(epoch_total_loss, label='Training loss')
    plt.plot(epoch_val_loss, label='Validation loss')
    plt.legend()
    plt.show()

    return best_valid_loss, test_loss_best_val

def train(train_loader, model, criterion, optimizer, epoch, args):
    total_loss = 0.0
    # switch to train mode
    model.train()
    stime = time.time()
    #Updating the parameters each iteration. (# of iterations = # batches)
    #Each iteration: 1)Forward Propogation 2)Compute Costs 3)Backpropagation 4)Update parameters
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        #???
        target = target.float()
        input = input.float()
        if args.is_cuda:
            target = target.cuda()
            input = input.cuda()

        #Forward pass to compute output
        output = model(input)
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

        for i, (input, target) in enumerate(val_loader):

            target = target.float()
            input = input.float()
            if args.is_cuda:
                target = target.cuda()
                input = input.cuda()

            # Forward pass to compute output
            output = model(input)
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


if __name__ == '__main__':
    args = parser.parse_args()
    best_valid_loss, test_loss_best_val = main(args)
    print("best_valid_loss:", best_valid_loss,  "test_loss_best_val:", test_loss_best_val)
