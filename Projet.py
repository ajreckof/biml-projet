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
import itertools
import csv
from time import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


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
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.linear1 = nn.Linear(400, 120)
        nn.init.uniform_(self.linear1.weight, -0.001, 0.001)
        self.linear2 = nn.Linear(120, 84)
        nn.init.uniform_(self.linear2.weight, -0.001, 0.001)
        self.linear3 = nn.Linear(84, 10)
        nn.init.uniform_(self.linear3.weight, -0.001, 0.001)

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

def write_results_to_csv(results, file_name, headers):
    f = open(file_name, 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write column names
    writer.writerow(headers)

    # write data to the csv file
    writer.writerows(results)

    # close the file
    f.close()

def generate_split(train_dataset, labels):

    # on génère des indices pour le set de validation et d'entrainement 
    train_indices, val_indices, _, _ = train_test_split(
        range(len(labels)),
        labels,
        stratify=labels,
        test_size=0.1,
    )

    # on génère les sous set à partir des indices
    train_split = Subset(train_dataset, train_indices)
    val_split = Subset(train_dataset, val_indices)

    return train_split, val_split

def test_model(model, test_loader):
    # test du modèle (on évalue la progression pendant l'apprentissage)
    acc = 0.
    # on lit toutes les données de test
    for x, t in test_loader:
        # on calcule la sortie du modèle
        y = model(x)
        # on regarde si la sortie est correcte
        acc += sum(torch.argmax(y, 1) == torch.argmax(t, 1))
    return float(acc/len(test_loader.dataset))

def train_model(model, train_loader, test_loader, eta=0.001, nb_epochs=10, verbose=True):
    
    # on initialise l'optimiseur
    loss_func = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(model.parameters(), lr=eta)

    acc = 0.
    start = time()
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
        if verbose:
            acc = test_model(model, test_loader)
            print(acc, time() - start)
    learning_time = time() - start

    if not verbose :
        acc = test_model(model, test_loader)

    return model, acc, learning_time


def grid_search_incrementing_factor(Model, possible_number_of_layers, possible_incrementing_factor, possible_batch_size, possible_learning_rates):
     # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)

    # on génère un set de validation
    train_split, val_split = generate_split(train_dataset, label_train)


    grid_parameter = itertools.product(
        possible_batch_size, 
        possible_learning_rates, 
        possible_number_of_layers, 
        possible_incrementing_factor
    )

    results = []
    best_model = None
    best_accuracy = 0

    for batch_size, learning_rate, number_of_layers, incrementing_factor in grid_parameter:

        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True)

        model, result, time = train_model(
            Model([10 * incrementing_factor ** (number_of_layers - i) for i in range(number_of_layers)]), 
            train_loader, 
            val_loader, 
            learning_rate, 
            verbose=False
        )
        if result > best_accuracy :
            best_accuracy = result
            best_model = model

        print(batch_size, learning_rate, number_of_layers, incrementing_factor, result, time)
        results.append([batch_size, learning_rate, number_of_layers, incrementing_factor, result, time])


    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return best_model, test_model(best_model, test_loader), results

def grid_search_first_layer_size(Model, possible_number_of_layers, possible_first_layer_size, possible_batch_size, possible_learning_rates):
     # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)

    # on génère un set de validation
    train_split, val_split = generate_split(train_dataset, label_train)


    grid_parameter = itertools.product(
        possible_batch_size, 
        possible_learning_rates, 
        possible_number_of_layers, 
        possible_first_layer_size
    )

    results = []
    best_model = None
    best_accuracy = 0

    for batch_size, learning_rate, number_of_layers, first_layer_size in grid_parameter:

        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True)

        model, result, time = train_model(
            Model([int(first_layer_size * (10/first_layer_size) ** (i/number_of_layers)) for i in range(number_of_layers)]), 
            train_loader, 
            val_loader,           
            learning_rate, 
            verbose=False
        )

        if result > best_accuracy :
            best_accuracy = result
            best_model = model

        print(batch_size, learning_rate, number_of_layers, first_layer_size, result, time)
        results.append([batch_size, learning_rate, number_of_layers, first_layer_size, result, time])
    
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return best_model, test_model(best_model, test_loader), results

def grid_search_shallow(Model, possible_neurones_num, possible_learning_rates):
     # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)

    # on génère un set de validation
    train_split, val_split = generate_split(train_dataset, label_train)

    #on génère les loader pour les set d'entrainement et de validation
    train_loader = DataLoader(train_split, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=5, shuffle=True)
    
    grid_parameter = itertools.product(possible_neurones_num, possible_learning_rates)

    results = []
    best_model = None
    best_accuracy = 0

    for N, eta in grid_parameter:
        model, result, time = train_model(
            Model(N),
            train_loader,
            val_loader, 
            eta,
            verbose=True)

        if result > best_accuracy :
            best_accuracy = result
            best_model = model
        
        print(eta, N, result, time)
        results.append([eta, N, result, time])
    return best_model, test_model(best_model, test_loader), results

if __name__ == '__main__':
    _, accuracy, results = grid_search_incrementing_factor(
        DeepNetwork, 
        [2, 3, 4], 
        [2,3,5],
        [5, 20, 50], 
        [10**-2, 10**-3, 10**-4]
    )
    print(accuracy)
    write_results_to_csv(
        results,
        "results_deep_incrementing_factor.csv",
        ["batch_size", "learning_rate", "number_of_layers", "incrementing_factor", "résultat", "temps"],
    )

    _, accuracy, results = grid_search_first_layer_size(
        DeepNetwork, 
        [2, 3, 4], 
        [100,200,500,1000],
        [5, 20, 50], 
        [10**-2, 10**-3, 10**-4]
    )
    print(accuracy)
    write_results_to_csv(
        results,
        "results_deep_first_layer_size.csv",
        ["batch_size", "learning_rate", "number_of_layers", "first_layer_size", "résultat", "temps"],
    )
