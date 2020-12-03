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

	idxs = list(range(r))
	train_idxs = random.sample(idxs, k=int(0.70*r))
	idxs = list(set(idxs)^set(train_idxs))
	valid_idxs = random.sample(idxs, k=int(0.15*r))
	test_idxs = list(set(idxs)^set(valid_idxs))

	train_data, valid_data, test_data = RPPA[train_idxs,:], RPPA[valid_idxs,:], RPPA[test_idxs,:]

	train_data, train_label = train_data[:,:101], train_data[:,101:]
	valid_data, valid_label = valid_data[:,:101], valid_data[:,101:]
	test_data, test_label = test_data[:,:101], test_data[:,101:]
	return train_data, train_label, valid_data, valid_label, test_data, test_label


if __name__ == '__main__':
	# df =  read_metab()
	data = read_RPPA() #just replace this with read_...
	print(data[0][0].shape, data[1][0].shape)
	dataset = TensorDataset(torch.tensor(data[0]), torch.tensor(data[1]))
	train_dataloader = DataLoader(dataset, batch_size=16)
	for data in train_dataloader:
		print(data[0].shape, data[1].shape)
		input()
