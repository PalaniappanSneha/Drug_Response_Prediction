import csv
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import random


torch.cuda.manual_seed(5)
torch.cuda.manual_seed_all(5)  # if you are using multi-GPU.
np.random.seed(5)  # Numpy module.
random.seed(5)  # Python random module.
torch.manual_seed(5)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def read_RPPA(path='Data/DNN/DNN_Input_RPPA.csv', cv = False):
	with open(path) as csvfile:
		RPPA_reader = csv.reader(csvfile)
		RPPA = np.array(list(RPPA_reader))
		RPPA = RPPA[1:,1:] #Ignoring the first column (model_id)
		RPPA = RPPA.astype(float)

	r,c = RPPA.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	#Select K-fold cv or simple holdout
	if cv:
		#Using both train and validation data for cross validation
		train_val_data, test_data = RPPA[np.concatenate([train_idxs, valid_idxs]),:], RPPA[test_idxs,:]
		train_val_data, train_val_label =  train_val_data[:,:101], train_val_data[:,101:]
		test_data, test_label = test_data[:,:101], test_data[:,101:]
		return train_val_data, train_val_label, test_data, test_label
	else:
		train_data, valid_data, test_data = RPPA[train_idxs,:], RPPA[valid_idxs,:], RPPA[test_idxs,:]

	#Differentiating features and labels
	train_data, train_label = train_data[:,:101], train_data[:,101:]
	valid_data, valid_label = valid_data[:,:101], valid_data[:,101:]
	test_data, test_label = test_data[:,:101], test_data[:,101:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label


def read_Meta(path='Data/DNN/DNN_Input_Metabolomics.csv', cv = False):
	with open(path) as csvfile:
		Meta_reader = csv.reader(csvfile)
		Meta = np.array(list(Meta_reader))
		Meta = Meta[1:,1:] #Ignoring the first column(label)
		Meta = Meta.astype(float)

	r,c = Meta.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	if cv:
		#Using both train and validation data for cross validation
		train_val_data, test_data = Meta[np.concatenate([train_idxs, valid_idxs]),:], Meta[test_idxs,:]
		train_val_data, train_val_label = train_val_data[:,:80], train_val_data[:,80:]
		test_data, test_label = test_data[:,:80], test_data[:,80:]
		return train_val_data, train_val_label, test_data, test_label
	else:
		train_data, valid_data, test_data = Meta[train_idxs,:], Meta[valid_idxs,:], Meta[test_idxs,:]

	#Differentiating features and labels
	train_data, train_label = train_data[:,:80], train_data[:,80:]
	valid_data, valid_label = valid_data[:,:80], valid_data[:,80:]
	test_data, test_label = test_data[:,:80], test_data[:,80:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label


def read_Expression(path='Data/DNN/DNN_Input_Expression.csv', cv = False):
	with open(path) as csvfile:
		exp_reader = csv.reader(csvfile)
		Exp = np.array(list(exp_reader))
		Exp = Exp[1:,1:] #Ignoring the first column(label)
		Exp = Exp.astype(float)

	r,c = Exp.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	if cv:
		#Using both train and validation data for cross validation
		train_val_data, test_data = Exp[np.concatenate([train_idxs, valid_idxs]),:], Exp[test_idxs,:]
		train_val_data, train_val_label =  train_val_data[:,:616], train_val_data[:,616:]
		test_data, test_label = test_data[:,:616], test_data[:,616:]
		return train_val_data, train_val_label, test_data, test_label
	else:
		train_data, valid_data, test_data = Exp[train_idxs,:], Exp[valid_idxs,:], Exp[test_idxs,:]

	#Differentiating features and labels
	train_data, train_label = train_data[:,:616], train_data[:,616:]
	valid_data, valid_label = valid_data[:,:616], valid_data[:,616:]
	test_data, test_label = test_data[:,:616], test_data[:,616:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label


def read_Mutations(path='Data/DNN/DNN_Input_Mutation.csv', cv = False):
	with open(path) as csvfile:
		mut_reader = csv.reader(csvfile)
		Mut = np.array(list(mut_reader))
		Mut = Mut[1:,1:] #Ignoring the first column(label)
		Mut = Mut.astype(float)

	r,c = Mut.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	if cv:
		#Using both train and validation data for cross validation
		train_val_data, test_data = Mut[np.concatenate([train_idxs, valid_idxs]),:], Mut[test_idxs,:]
		train_val_data, train_val_label =  train_val_data[:,:1040], train_val_data[:,1040:]
		test_data, test_label = test_data[:,:1040], test_data[:,1040:]
		return train_val_data, train_val_label, test_data, test_label
	else:
		train_data, valid_data, test_data = Mut[train_idxs,:], Mut[valid_idxs,:], Mut[test_idxs,:]

	#Differentiating features and labels
	train_data, train_label = train_data[:,:1040], train_data[:,1040:]
	valid_data, valid_label = valid_data[:,:1040], valid_data[:,1040:]
	test_data, test_label = test_data[:,:1040], test_data[:,1040:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label


def read_CNV(path='Data/DNN/DNN_Input_CNV.csv', cv = False):
	with open(path) as csvfile:
		CNV_reader = csv.reader(csvfile)
		CNV = np.array(list(CNV_reader))
		CNV = CNV[1:,1:] #Ignoring the first column(label)
		CNV = CNV.astype(float)

	r,c = CNV.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	if cv:
		#Using both train and validation data for cross validation
		train_val_data, test_data = CNV[np.concatenate([train_idxs, valid_idxs]),:], CNV[test_idxs,:]
		train_val_data, train_val_label =  train_val_data[:,:88], train_val_data[:,88:]
		test_data, test_label = test_data[:,:88], test_data[:,88:]
		return train_val_data, train_val_label, test_data, test_label
	else:
		train_data, valid_data, test_data = CNV[train_idxs,:], CNV[valid_idxs,:], CNV[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:88], train_data[:,88:]
	valid_data, valid_label = valid_data[:,:88], valid_data[:,88:]
	test_data, test_label = test_data[:,:88], test_data[:,88:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label

def read_miRNA(path='Data/DNN/DNN_Input_miRNA.csv', cv = False):
	with open(path) as csvfile:
		miRNA_reader = csv.reader(csvfile)
		miRNA = np.array(list(miRNA_reader))
		miRNA = miRNA[1:,1:] #Ignoring the first column(label)
		miRNA = miRNA.astype(float)

	r,c = miRNA.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	if cv:
		#Using both train and validation data for cross validation
		train_val_data, test_data = miRNA[np.concatenate([train_idxs, valid_idxs]),:], miRNA[test_idxs,:]
		train_val_data, train_val_label =  train_val_data[:,:197], train_val_data[:,197:]
		test_data, test_label = test_data[:,:197], test_data[:,197:]
		return train_val_data, train_val_label, test_data, test_label
	else:
		train_data, valid_data, test_data = miRNA[train_idxs,:], miRNA[valid_idxs,:], miRNA[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:197], train_data[:,197:]
	valid_data, valid_label = valid_data[:,:197], valid_data[:,197:]
	test_data, test_label = test_data[:,:197], test_data[:,197:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label

def read_combined(path='Data/DNN/DNN_Combined_Input_2.csv',cv = False):
	with open(path) as csvfile:
		comb_reader = csv.reader(csvfile)
		comb = np.array(list(comb_reader))
		comb = comb[1:,1:] #Ignoring the first column(label)
		comb = comb.astype(float)

	r,c = comb.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.80*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.10*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	#Need to verify
	if cv:
		train_val_data, test_data = comb[np.concatenate([train_idxs, valid_idxs]),:], comb[test_idxs,:]
		train_val_data, train_val_label =  train_val_data[:,:2122], train_val_data[:,2122:]
		test_data, test_label = test_data[:,:2122], test_data[:,2122:]

		return train_val_data, train_val_label, test_data, test_label

	else:
		train_data, valid_data, test_data = comb[train_idxs,:], comb[valid_idxs,:], comb[test_idxs,:]
	#Differentiating features and labels
	train_data_RPPA,  train_data_miRNA, train_data_Meta, train_data_Mut, train_data_Exp, train_data_CNV,train_label = train_data[:,:101],train_data[:,101:298],train_data[:,298:378],train_data[:,378:1418],train_data[:,1418:2034],train_data[:,2034:2122],train_data[:,2122:]
	valid_data_RPPA, valid_data_miRNA, valid_data_Meta,  valid_data_Mut, valid_data_Exp, valid_data_CNV,valid_label = valid_data[:,:101],valid_data[:,101:298],valid_data[:,298:378],valid_data[:,378:1418],valid_data[:,1418:2034],valid_data[:,2034:2122],valid_data[:,2122:]
	test_data_RPPA, test_data_miRNA, test_data_Meta,  test_data_Mut, test_data_Exp, test_data_CNV,test_label = test_data[:,:101],test_data[:,101:298],test_data[:,298:378],test_data[:,378:1418],test_data[:,1418:2034],test_data[:,2034:2122],test_data[:,2122:]

	return (torch.tensor(train_data_RPPA), torch.tensor(train_data_miRNA), torch.tensor(train_data_Meta), torch.tensor(train_data_Mut), torch.tensor(train_data_Exp), torch.tensor(train_data_CNV),torch.tensor(train_label)),\
	 (torch.tensor(valid_data_RPPA), torch.tensor(valid_data_miRNA), torch.tensor(valid_data_Meta), torch.tensor(valid_data_Mut), torch.tensor(valid_data_Exp), torch.tensor(valid_data_CNV),torch.tensor(valid_label)), \
	 (torch.tensor(test_data_RPPA), torch.tensor(test_data_miRNA), torch.tensor(test_data_Meta), torch.tensor(test_data_Mut), torch.tensor(test_data_Exp), torch.tensor(test_data_CNV),torch.tensor(test_label))
#read_combined for 5 datasets
# def read_combined(path='Data/DNN/DNN_Combined_Input_1.csv',cv = False):
# 	with open(path) as csvfile:
# 		comb_reader = csv.reader(csvfile)
# 		comb = np.array(list(comb_reader))
# 		comb = comb[1:,1:] #Ignoring the first column(label)
# 		comb = comb.astype(float)
#
# 	r,c = comb.shape
# 	#Split data into train,validation,test data (using indexes)
# 	idxs = list(range(r))
# 	train_idxs = random.sample(idxs, k=int(0.80*r))
# 	idxs = list(set(idxs)^set(train_idxs))
# 	valid_idxs = random.sample(idxs, k=int(0.20*r))
# 	test_idxs = list(set(idxs)^set(valid_idxs))
#
# 	#Need to verify
# 	if cv:
# 		train_val_data, test_data = comb[np.concatenate([train_idxs, valid_idxs]),:], comb[test_idxs,:]
# 		train_val_data, train_val_label =  train_val_data[:,:1925], train_val_data[:,1925:]
# 		test_data, test_label = test_data[:,:1925], test_data[:,1925:]
#
# 		return train_val_data, train_val_label, test_data, test_label
#
# 	else:
# 		train_data, valid_data, test_data = comb[train_idxs,:], comb[valid_idxs,:], comb[test_idxs,:]
# 	#Differentiating features and labels
# 	train_data_RPPA, train_data_Meta, train_data_Mut, train_data_Exp, train_data_CNV,train_label = train_data[:,:101],train_data[:,101:181],train_data[:,181:1221],train_data[:,1221:1837],train_data[:,1837:1925],train_data[:,1925:]
# 	valid_data_RPPA, valid_data_Meta, valid_data_Mut, valid_data_Exp, valid_data_CNV,valid_label = valid_data[:,:101],valid_data[:,101:181],valid_data[:,181:1221],valid_data[:,1221:1837],valid_data[:,1837:1925],valid_data[:,1925:]
# 	test_data_RPPA, test_data_Meta, test_data_Mut, test_data_Exp, test_data_CNV,test_label = test_data[:,:101],test_data[:,101:181],test_data[:,181:1221],test_data[:,1221:1837],test_data[:,1837:1925],test_data[:,1925:]
#
# 	return (torch.tensor(train_data_RPPA), torch.tensor(train_data_Meta), torch.tensor(train_data_Mut), torch.tensor(train_data_Exp), torch.tensor(train_data_CNV),torch.tensor(train_label)),\
# 	 (torch.tensor(valid_data_RPPA), torch.tensor(valid_data_Meta), torch.tensor(valid_data_Mut), torch.tensor(valid_data_Exp), torch.tensor(valid_data_CNV),torch.tensor(valid_label)), \
# 	 (torch.tensor(test_data_RPPA), torch.tensor(test_data_Meta), torch.tensor(test_data_Mut), torch.tensor(test_data_Exp), torch.tensor(test_data_CNV),torch.tensor(test_label))
#

if __name__ == '__main__':
	#read_RPPA returns 6 things
	data = read_miRNA() #just replace this with read_...
	#Check training features and label shape
	print(data[0][0].shape, data[1][0].shape)
	#Convert to tensors
	dataset = TensorDataset(torch.tensor(data[0]), torch.tensor(data[1]))
	#Creating batches and making datasets iterable
	train_dataloader = DataLoader(dataset, batch_size=16)
	#Iterate over the data
	for data in train_dataloader:
		#prints shape of one batch of data (16,101),(16,24)
		print(data[0].shape, data[1].shape)
		#comment this line to see how many batches are created
		input()
