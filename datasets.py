import csv
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import random

# def read_metab(path='../Data/PCA/Metabolomics_PCA_with_label.csv'):
# 	metab = pd.read_csv(path)
# 	return metab.to_numpy()

# def read_RPPA(path='../Data/PCA/RPPA_PCA_with_label.csv'):
# 	RPPA = pd.read_csv(path)
# 	RPPA.to_numpy()
# 	return RPPA


def read_RPPA(path='Data/DNN/DNN_Input_RPPA.csv'):
	with open(path) as csvfile:
		RPPA_reader = csv.reader(csvfile)
		RPPA = np.array(list(RPPA_reader))
		RPPA = RPPA[1:,1:] #Ignoring the first column(label)
		RPPA = RPPA.astype(float)

	r,c = RPPA.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.70*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.15*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	train_data, valid_data, test_data = RPPA[train_idxs,:], RPPA[valid_idxs,:], RPPA[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:101], train_data[:,101:]
	valid_data, valid_label = valid_data[:,:101], valid_data[:,101:]
	test_data, test_label = test_data[:,:101], test_data[:,101:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label

def read_Meta(path='Data/DNN/DNN_Input_Metabolomics.csv'):
	with open(path) as csvfile:
		Meta_reader = csv.reader(csvfile)
		Meta = np.array(list(Meta_reader))
		Meta = Meta[1:,1:] #Ignoring the first column(label)
		Meta = Meta.astype(float)

	r,c = Meta.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.70*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.15*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	train_data, valid_data, test_data = Meta[train_idxs,:], Meta[valid_idxs,:], Meta[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:80], train_data[:,80:]
	valid_data, valid_label = valid_data[:,:80], valid_data[:,80:]
	test_data, test_label = test_data[:,:80], test_data[:,80:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label

def read_Expression(path='Data/DNN/DNN_Input_Expression.csv'):
	with open(path) as csvfile:
		exp_reader = csv.reader(csvfile)
		Exp = np.array(list(exp_reader))
		Exp = Exp[1:,1:] #Ignoring the first column(label)
		Exp = Exp.astype(float)

	r,c = Exp.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.70*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.15*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	train_data, valid_data, test_data = Exp[train_idxs,:], Exp[valid_idxs,:], Exp[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:616], train_data[:,616:]
	valid_data, valid_label = valid_data[:,:616], valid_data[:,616:]
	test_data, test_label = test_data[:,:616], test_data[:,616:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label

def read_Mutations(path='Data/DNN/DNN_Input_Mutation.csv'):
	with open(path) as csvfile:
		mut_reader = csv.reader(csvfile)
		Mut = np.array(list(mut_reader))
		Mut = Mut[1:,1:] #Ignoring the first column(label)
		Mut = Mut.astype(float)

	r,c = Mut.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.70*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.15*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	train_data, valid_data, test_data = Mut[train_idxs,:], Mut[valid_idxs,:], Mut[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:1040], train_data[:,1040:]
	valid_data, valid_label = valid_data[:,:1040], valid_data[:,1040:]
	test_data, test_label = test_data[:,:1040], test_data[:,1040:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label

def read_CNV(path='Data/DNN/DNN_Input_CNV.csv'):
	with open(path) as csvfile:
		CNV_reader = csv.reader(csvfile)
		CNV = np.array(list(CNV_reader))
		CNV = CNV[1:,1:] #Ignoring the first column(label)
		CNV = CNV.astype(float)

	r,c = CNV.shape
	#Split data into train,validation,test data (using indexes)
	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.70*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.15*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	train_data, valid_data, test_data = CNV[train_idxs,:], CNV[valid_idxs,:], CNV[test_idxs,:]
	#Differentiating features and labels
	train_data, train_label = train_data[:,:88], train_data[:,88:]
	valid_data, valid_label = valid_data[:,:88], valid_data[:,88:]
	test_data, test_label = test_data[:,:88], test_data[:,88:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label


if __name__ == '__main__':
	# df =  read_metab()
	#read_RPPA returns 6 things
	data = read_RPPA() #just replace this with read_...
	#Check training features and label shape
	print(data[0][0].shape, data[1][0].shape)
	#Convert to tensors?
	dataset = TensorDataset(torch.tensor(data[0]), torch.tensor(data[1]))
	#Creating batches and making datasets iterable
	train_dataloader = DataLoader(dataset, batch_size=16)
	for data in train_dataloader:
		#prints shape of one batch of data (16,101),(16,24)
		print(data[0].shape, data[1].shape)
		#comment this line to see how many batches are created
		input()
