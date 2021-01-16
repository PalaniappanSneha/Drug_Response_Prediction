def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from model import Net, Net_combined,  Net_CNN
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
from sklearn.metrics import r2_score, f1_score

torch.manual_seed(5)
torch.cuda.manual_seed(5)
torch.cuda.manual_seed_all(5)  # if you are using multi-GPU.
np.random.seed(5)  # Numpy module.
# random.seed(10)  # Python random module.
torch.manual_seed(5)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Drug Response Prediction')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--esthres', type=int, default=10, help='')
# parser.add_argument('--est', type=int, default=30, help='early_stopping_threshold')

parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')
parser.add_argument('--data', type=str, default='RPPA', help='')
parser.add_argument('--model', type=str, default='Net', help='')
parser.add_argument('--expr_dir', type=str, default="experiments/", help='path/to/save_dir')
parser.add_argument('--cv', action='store_true', help='cross validation') #--cv
# parser.add_argument('--num_param', type=int, default =101)

parser.add_argument('--out_embed', type=int, default=200)
parser.add_argument('--out_lay2', type=int, default =128) #comment for one layer
parser.add_argument('--out_lay3', type=int, default =64)
parser.add_argument('--output_dim',type=int, default=24)
parser.add_argument('--dropout',type=float, default=0)

def calc_r2 (x,y):
    total = 0
    for i in range(24):
        temp= r2_score(x[:,i],y[:,i])
        total += temp
    total /= 24
    return total

def calc_accuracy(x,y):
    correct = np.sum(x == y)
    total = len(x)
    accuracy = (correct*100/total)

    return accuracy

def top_k(x,y):
    total = len(x)
    correct = 0
    for i in range(len(x)):
        if(x[i] in y[i].numpy()): #if pred in top actual values
            correct += 1
    topkaccuracy = (correct*100/total)
    return topkaccuracy

def main(args, train_data,valid_data,test_data):

    ### init training and val stuff ###
    #Loss Function
    criterion = nn.MSELoss()

    train_data,valid_data,test_data = read_combined()

    args.num_param = 101
    args.out_embed = 101
    #args.dropout = 0.1
    model_RPPA = Net(args)
    # model_RPPA = Net_CNN(args)
    model_RPPA = load_checkpoint('experiments/RPPA/model_best.pth.tar', model_RPPA)

    args.num_param = 197
    args.out_embed = 197
    model_miRNA = Net(args)
    # model_miRNA = Net_CNN(args)
    model_miRNA = load_checkpoint('experiments/miRNA/model_best.pth.tar', model_miRNA) #create folder

    args.num_param = 80
    args.out_embed = 80
    model_Meta = Net(args)
    # model_Meta = Net_CNN(args)
    model_Meta = load_checkpoint('experiments/Meta/model_best.pth.tar', model_Meta)

    args.num_param = 1040
    args.out_embed = 1040
    model_Mut = Net(args)
    # model_Mut = Net_CNN(args)
    model_Mut = load_checkpoint('experiments/Mut/model_best.pth.tar', model_Mut)

    args.num_param = 616
    args.out_embed = 616
    model_Exp = Net(args)
    # model_Exp = Net_CNN(args)
    model_Exp = load_checkpoint('experiments/Exp/model_best.pth.tar', model_Exp)

    args.num_param = 88
    args.out_embed = 88
    model_CNV = Net(args)
    # model_CNV = Net_CNN(args)
    model_CNV = load_checkpoint('experiments/CNV/model_best.pth.tar', model_CNV)


    args.is_cuda = torch.cuda.is_available()

    model = Net_combined(args, model_RPPA, model_miRNA, model_Meta, model_Mut, model_Exp, model_CNV)
    if args.is_cuda:
        model = model.cuda()

    #Adam optimizer
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer = torch.optim.Adam(model.output.parameters(), args.lr, weight_decay=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)

    train_loader = torch.utils.data.DataLoader(TensorDataset(train_data[0],train_data[1], train_data[2],train_data[3],train_data[4],train_data[5],train_data[6]), batch_size=args.batchSize, shuffle = True,num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(TensorDataset(valid_data[0],valid_data[1], valid_data[2],valid_data[3],valid_data[4],valid_data[5],valid_data[6]), batch_size=args.batchSize, shuffle = True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_data[0],test_data[1], test_data[2],test_data[3],test_data[4],test_data[5], test_data[6]), batch_size=args.batchSize, shuffle = True, num_workers=0, pin_memory=True)

    ### print options ###

    all_train_loss, all_valid_loss, all_test_loss = [],[],[]
    all_train_TT = []
    all_train_r_square= []
    all_valid_r_square =[]
    all_test_r_square =[]
    all_accuracy, all_topkaccuracy, all_f1 = [],[],[]
    all_tmp1, all_tmp2 = [],[]

    #For Early Stopping
    best_valid_loss = 99999
    test_loss_best_val = 99999
    test_rsquare_best_valid = -999999999
    max_r2_test = -99999999999
    max_r2_train = -99999999

    count = 0


    for epoch in range(0,args.epoch):

        # train for one epoch
        # returns total_loss, TT and r2 for 1 epoch
        epoch_total_loss, TT, r_square_train, tmp1 = train(train_loader, model, criterion, optimizer, epoch, args)

        #loss,r2 and TT for total epochs
        all_train_loss.append(epoch_total_loss)
        all_train_r_square.append(r_square_train)
        all_train_TT.append(TT)
        all_tmp1.append(tmp1)

        # evaluate on validation set
        epoch_val_loss, r_square_val, accuracy, topkaccuracy, f1, tmp2  = validate(val_loader, model, criterion, args)
        all_valid_loss.append(epoch_val_loss)
        all_valid_r_square.append(r_square_val )

        # evaluate on test set
        epoch_test_loss, r_square_test, epoch_acc, epoch_topk, epoch_f1, tmp2= validate(test_loader, model, criterion, args,test_flag =True)
        all_test_loss.append(epoch_test_loss)
        all_test_r_square.append(r_square_test )
        all_accuracy.append(epoch_acc)
        all_topkaccuracy.append(epoch_topk)
        all_f1.append(epoch_f1)
        all_tmp2.append(tmp2)

        test_r_square_best_valid = r_square_test
        #Early Stopping
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }
        #Get best r2 values
        if(r_square_train > max_r2_train):
            max_r2_train = r_square_train
        if(r_square_test > max_r2_test):
            max_r2_test = r_square_test

        if(epoch_val_loss < best_valid_loss):
            count = 0
            best_valid_loss = epoch_val_loss
            test_loss_best_val = epoch_test_loss
            test_r_square_best_valid = r_square_test

            best_train_loss = epoch_total_loss

            best_accuracy = epoch_acc
            best_topk = epoch_topk
            best_f1 = epoch_f1

            best_tmp1 = tmp1
            best_tmp2 = tmp2


            save_checkpoint(state, True, args)
        else:
            count = count + 1
            if(count >= args.esthres):
                break

    #Calc mean r_square
    avg_test_r_square = np.mean(all_test_r_square)
    avg_valid_r_square = np.mean(all_valid_r_square)
    avg_train_r_square = np.mean(all_train_r_square)
    #Calc std r_square
    std_test_r_square = np.std(all_test_r_square)
    std_valid_r_square = np.std(all_valid_r_square)
    std_train_r_square = np.std(all_train_r_square)

    #plotting code
    #plotting the training and validation loss
    plt.clf() #clear
    plt.plot(all_train_loss, label='Training loss')
    plt.plot(all_valid_loss, label='Validation loss')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('No. of epochs')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(os.path.join(args.expr_dir, 'combined_x14.png'))


    return best_valid_loss, test_loss_best_val, avg_test_r_square, std_test_r_square, max_r2_train, max_r2_test,best_train_loss,best_accuracy, best_topk, best_f1, best_tmp1, best_tmp2

def train(train_loader, model, criterion, optimizer, epoch, args):
    total_loss = 0.0
    # switch to train mode
    model.train()

    pred_values = np.empty((0,24))
    target_values = np.empty((0,24))
    stime = time.time()

    #Updating the parameters each iteration. (# of iterations = # batches)
    #Each iteration: 1)Forward Propogation 2)Compute Costs 3)Backpropagation 4)Update parameters
    for i, (input1,input2,input3,input4,input5,input6, target) in enumerate(train_loader):
        target = target.float()
        input1,input2,input3,input4,input5,input6 = input1.float(),input2.float(),input3.float(),input4.float(),input5.float(),input6.float()

        if args.is_cuda:
            target = target.cuda()
            input1,input2,input3,input4,input5,input6 = input1.cuda(),input2.cuda(),input3.cuda(),input4.cuda(),input5.cuda(),input6.cuda()

        #Forward pass to compute output
        output = model(input1,input2,input3,input4,input5,input6)
        #storing predicted and target values
        pred_values=np.concatenate((pred_values,output.cpu().detach().numpy()), axis=0)
        target_values = np.concatenate((target_values,target.cpu().detach().numpy()) ,axis=0)
        #Calculate Loss: MSE
        loss = criterion(output, target)
        tmp_1 = torch.mean((output[:,23]-target[:,23])**2)
        #Adding loss for current iteration into total_loss
        total_loss += loss.detach().item() #total_loss += loss
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
    #calc r2
    # r_square =  calc_r2(pred_values, target_values)???
    r_square =  calc_r2(pred_values, target_values)
    #Find best drug
    best_drug_target = np.argmin(target_values, axis = -1) #Returns the indices of the minimum values along an axis
    best_drug_predicted = np.argmin(pred_values, axis = -1)
    #Find top 3 drugs
    best_drug_target_top3 = torch.topk(torch.tensor(target_values), k=3, largest = False, dim=-1)[1]

    accuracy = calc_accuracy(best_drug_target,best_drug_predicted)
    topkaccuracy = top_k(best_drug_predicted, best_drug_target_top3) #? (best_drug_target_top3, best_drug_predicted)
    f1 = f1_score(best_drug_predicted,best_drug_target, average = 'micro')

    if args.verbose:
        print('Epoch: [{0}]\t'
          'Training Loss {loss:.3f}\t'
          'Time: {time:.2f}\t'
          'r_square: {r2}\t'
          'accuracy: {acc}\t'
          'topkaccuracy:{kacc}\t'
          'f1_score: {f1}'
          'tmp_1: {temp1}\t'.format(
           epoch, loss=total_loss, time= TT, r2=r_square, acc=accuracy, kacc = topkaccuracy, f1 = f1, temp1 = tmp_1))

    return total_loss, TT, r_square, tmp_1

def validate(val_loader, model, criterion, args, test_flag=False):

    # switch to evaluate mode
    model.eval()

    total_loss = 0.0
    pred_values = np.empty((0,24))
    target_values = np.empty((0,24))
    #Prevent tracking history,for validation.
    with torch.no_grad():

        for i, (input1,input2,input3,input4,input5,input6, target) in enumerate(val_loader):

            target = target.float()
            input1,input2,input3,input4,input5,input6 = input1.float(),input2.float(),input3.float(),input4.float(),input5.float(),input6.float()
            if args.is_cuda:
                target = target.cuda()
                input1,input2,input3,input4,input5,input6 = input1.cuda(),input2.cuda(),input3.cuda(),input4.cuda(),input5.cuda(), input6.cuda()

            # Forward pass to compute output
            output = model(input1,input2,input3,input4,input5,input6)
            pred_values=np.concatenate((pred_values,output.cpu().detach().numpy()), axis=0)
            target_values = np.concatenate((target_values,target.cpu().detach().numpy()) ,axis=0)
            #Calculate Loss: MSE
            loss = criterion(output, target)
            tmp_2 = torch.mean((output[:,23]-target[:,23])**2)
            total_loss += loss.item() #total_loss += loss


    total_loss =  total_loss/(i+1)
    r_square =  calc_r2(pred_values, target_values)

    #Find best drug
    best_drug_target = np.argmin(target_values, axis = -1) #Returns the indices of the minimum values along an axis
    best_drug_predicted = np.argmin(pred_values, axis = -1)
    #Find top 3 drugs
    best_drug_target_top3 = torch.topk(torch.tensor(target_values), k=3, largest = False, dim=-1)[1]

    accuracy = calc_accuracy(best_drug_target,best_drug_predicted)
    topkaccuracy = top_k(best_drug_predicted, best_drug_target_top3) #? (best_drug_target_top3, best_drug_predicted)
    f1 = f1_score(best_drug_predicted,best_drug_target, average = 'micro')

    #print options
    if test_flag:
        txt = 'Test'
    else:
        txt = 'Val'

    if args.verbose:
        print('{type}: \t'
          'Loss {loss:.4f}\t'.format(type=txt,loss=total_loss))

    return total_loss, r_square, accuracy, topkaccuracy, f1, tmp_2

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
        #train_val_data,test_data - returns 5 features and 1 label;
        train_val_data, train_val_label, test_data, test_label = read_combined(cv=args.cv)
    else:
        train_data,valid_data,test_data = read_combined(cv=args.cv)

    #print options
    if args.cv:
        args.verbose = False
    else:
        args.verbose = True

    if args.cv:
        #2) Perform kfold cross-validation
        # kf = KFold(n_splits=3)
        kf = KFold(n_splits=3, shuffle=True, random_state=5)
        kf.get_n_splits(train_val_data)

        best_valid_results = []
        test_loss_best_val_results = []
        test_r_square_cv = []
        std_test_r_square_cv = []

        #new (using best values of r2)
        r2_log = []

        #acc,topk,f1
        acc_log = []
        topk_log = []
        f1_log = []

        test_data_RPPA, test_data_miRNA, test_data_Meta,  test_data_Mut, test_data_Exp, test_data_CNV = test_data[:,:101],test_data[:,101:298],test_data[:,298:378],test_data[:,378:1418],test_data[:,1418:2034],test_data[:,2034:2122]
        test_data =  (torch.tensor(test_data_RPPA), torch.tensor(test_data_miRNA), torch.tensor(test_data_Meta), torch.tensor(test_data_Mut), torch.tensor(test_data_Exp), torch.tensor(test_data_CNV),torch.tensor(test_label))
        #In this case text_index is my val_index
        for train_index, test_index in kf.split(train_val_data):
            # print(train_index)
            # print(test_index)
            train_data, val_data = train_val_data[train_index], train_val_data[test_index]
            train_label, val_label = train_val_label[train_index], train_val_label[test_index]
            #Differentiating features and labels
            train_data_RPPA,  train_data_miRNA, train_data_Meta, train_data_Mut, train_data_Exp, train_data_CNV = train_data[:,:101],train_data[:,101:298],train_data[:,298:378],train_data[:,378:1418],train_data[:,1418:2034],train_data[:,2034:2122]
            valid_data_RPPA, valid_data_miRNA, valid_data_Meta,  valid_data_Mut, valid_data_Exp, valid_data_CNV = val_data[:,:101],val_data[:,101:298],val_data[:,298:378],val_data[:,378:1418],val_data[:,1418:2034],val_data[:,2034:2122]

            train_data =  (torch.tensor(train_data_RPPA), torch.tensor(train_data_miRNA), torch.tensor(train_data_Meta), torch.tensor(train_data_Mut), torch.tensor(train_data_Exp), torch.tensor(train_data_CNV),torch.tensor(train_label))
            valid_data =  (torch.tensor(valid_data_RPPA), torch.tensor(valid_data_miRNA), torch.tensor(valid_data_Meta), torch.tensor(valid_data_Mut), torch.tensor(valid_data_Exp), torch.tensor(valid_data_CNV),torch.tensor(val_label))

            best_valid_loss, test_loss_best_val, avg_test_r_square, std_test_r_square, max_r2_train, max_r2_test,best_train_loss, best_accuracy, best_topk, best_f1, best_tmp1, best_tmp2   = main(args, train_data,valid_data,test_data)

            best_valid_results.append(best_valid_loss)
            test_loss_best_val_results.append(test_loss_best_val)
            #new
            std_mse_test = np.std(test_loss_best_val_results)

            test_r_square_cv.append(avg_test_r_square)
            std_test_r_square_cv.append(std_test_r_square)

            #newly added (using best values of r2, max_r2_test)
            r2_log.append(max_r2_test)

            #acc,topk,f1
            acc_log.append(best_accuracy)
            topk_log.append(best_topk)
            f1_log.append(best_f1)

        #Get avg. scores obtained across the k-folds
        best_valid_average = np.mean(best_valid_results)
        test_loss_best_val_average = np.mean(test_loss_best_val_results)

        test_r2_average_cv = np.mean(test_r_square_cv)
        test_r2_std_cv = np.mean(std_test_r_square_cv)

        #newly added (using best values of r2)
        mean_r2_test = np.mean(r2_log)
        std_r2_test = np.std(r2_log)

        acc_avg = np.mean(acc_log)
        topk_avg = np.mean(topk_log)
        f1_avg = np.mean(f1_log)

        # print("best_valid_loss_CV_avg:", best_valid_average,  "test_loss_best_val_CV_avg:", test_loss_best_val_average, "best_test_r2_average:", test_r2_average_cv, "best_test_r2_std:", test_r2_std_cv)
        print("best_valid_loss_CV_avg:", best_valid_average,  "test_loss_best_val_CV_avg:", test_loss_best_val_average,"best_test_r2_average:", test_r2_average_cv, "best_test_r2_std:", test_r2_std_cv, "mean_r2_test", mean_r2_test, "std_r2_test", std_r2_test, "std_mse",std_mse_test, "acc_avg",acc_avg,"topk_avg",topk_avg,"f1_avg",f1_avg)
    else:
        best_valid_loss, test_loss_best_val, avg_test_r_square, std_test_r_square, max_r2_train, max_r2_test,best_train_loss, best_accuracy, best_topk, best_f1, best_tmp1, best_tmp2  = main(args, train_data,valid_data,test_data)
        # print("best_valid_loss:", best_valid_loss,  "test_loss_best_val:", test_loss_best_val, "best_test_r2_average:", avg_test_r_square, "best_test_r2_std:", std_test_r_square)
        print("best_train_loss", best_train_loss,"best_valid_loss:", best_valid_loss,  "test_loss_best_val:", test_loss_best_val, "best_test_r2_average:", avg_test_r_square, "best_test_r2_std:", std_test_r_square, "max_r2_train:", max_r2_train, "max_r2_test", max_r2_test, "Drug_train", best_tmp1, "Drug_test", best_tmp2 )

    # best_valid_loss, test_loss_best_val = main(args)
    # print("best_valid_loss:", best_valid_loss,  "test_loss_best_val:", test_loss_best_val)
