# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import gzip
import numpy
import torch
from time import time


class ShallowNetwork(nn.Module):
	def __init__(self, N) -> None:
		super().__init__()
		self.linear1 = nn.Linear(784, N)
		nn.init.uniform_(self.linear1.weight, -0.001, 0.001)
		self.linear2 = nn.Linear(N, 10)
		nn.init.uniform_(self.linear2.weight, -0.001, 0.001)

	def forward(self, x):
		x = self.linear1(x)
		x = F.relu(x)
		return self.linear2(x)


class DeepNetwork(nn.Module):
	def __init__(self, N_list) -> None:
		super().__init__()
		N_list = [784] + N_list + [10]
		self.nb_couche = len(N_list) - 1
		self.linears = nn.ModuleList(
			[nn.Linear(N_list[i], N_list[i+1]) for i in range(self.nb_couche)])

	def forward(self, x):
		for i in range(self.nb_couche):
			x = self.linears[i](x)
			if i + 1 != self.nb_couche:
				x = F.relu(x)
		return x

class LeNet5(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(6),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		self.linear1 = nn.Linear(400, 120)
		nn.init.uniform_(self.linear1.weight, -0.001, 0.001)
		self.linear2 = nn.Linear(120, 84)
		nn.init.uniform_(self.linear2.weight, -0.001, 0.001)
		self.linear3 = nn.Linear(84, 10)
		nn.init.uniform_(self.linear3.weight, -0.001, 0.001)
		
	def forward(self, x):
		x = x.view(x.size(0),1,28,28)
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.reshape(x.size(0), -1)
		x = self.linear1(x)
		x = F.relu(x)
		x = self.linear2(x)
		x = F.relu(x)
		x = self.linear3(x)
		return x

def train_model(model, verbose=True):
	start = time()
	batch_size = 5  # nombre de données lues à chaque fois
	nb_epochs = 10  # nombre de fois que la base de données sera lue
	eta = 0.001  # taux d'apprentissage

	# on lit les données
	((data_train, label_train), (data_test, label_test)
	 ) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(
		data_train, label_train)
	test_dataset = torch.utils.data.TensorDataset(
		data_test, label_test)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=1, shuffle=False)

	# on initialise le modèle et ses poids

	# on initiliase l'optimiseur
	loss_func = torch.nn.MSELoss(reduction='sum')
	optim = torch.optim.SGD(model.parameters(), lr=eta)
	acc = 0.

	if verbose:
		print(time() - start)
	for n in range(nb_epochs):
		# on lit toutes les données d'apprentissage
		for x, t in train_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on met à jour les poids
			loss = loss_func(t, y)
			loss.backward()
			optim.step()
			optim.zero_grad()

		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for x, t in test_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on regarde si la sortie est correcte
			acc += torch.argmax(y, 1) == torch.argmax(t, 1)
		# on affiche le pourcentage de bonnes réponses
		if verbose:
			print(acc/data_test.shape[0], time() - start)
	return acc/data_test.shape[0]

if __name__ == '__main__':
	train_model(model=LeNet5())
