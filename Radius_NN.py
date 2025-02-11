# Ali Özçelik 2306579

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

train_set = pd.read_csv('/Users/desidero/Desktop/Dersler/IE460/assignment 3/train_set.csv')
test_set = pd.read_csv('/Users/desidero/Desktop/Dersler/IE460/assignment 3/test_set.csv')


class RadiusKNN:
    
    def __init__(self, radius, train_set, test_set):
        self.radius = radius
        self.closest_indexes = []
        self.train_set = train_set.iloc[:, 1:].values
        self.test_set = test_set.iloc[:, 1:].values
        self.split()
        self.test_accuracy, self.train_accuracy = self.give_accuracy()
        
    
    def split(self):
        self.train_attributes = self.train_set[:, :-1]
        self.train_labels = self.train_set[:, -1]
        self.test_attributes = self.test_set[:, :-1]
        self.test_labels = self.test_set[:, -1]
        
        
    def distance(self, vector1, vector2):
        dist = np.linalg.norm(vector1 - vector2)
        return dist 
    
    # counts only 1s 
    def count(self, label_list):
        c = 0
        for i in label_list:
            if i == 1:
                c+= 1
        if c >= (len(label_list)/2):
            return True
        
        return False
    
    # data_point is numpy array 
    # as output, closest k data point indexes are provided in a list
    def closest_neighbours(self, data_point, data_point_index):
        distances = [self.distance(data_point, self.train_attributes[i, :]) for i in range(len(self.train_attributes))]
        closest_distances = [distance for distance in distances if distance <= self.radius]
        closest_indexes = [distances.index(i) for i in closest_distances]
        return closest_distances, closest_indexes

    # labels are numpy array
    # dataset is numpy array
    def predict(self, dataset, data_point_index):
        data_point = dataset[data_point_index, :]
        closest_distances, closest_indexes = self.closest_neighbours(data_point, data_point_index)
        closest_labels = [self.train_labels[i] for i in closest_indexes]
        self.closest_indexes.append(closest_indexes)
        if self.count(closest_labels):
            return 1
        return 0


    # labels are numpy array
    def give_accuracy(self):
        predictions_test = [self.predict(self.test_attributes[:, :], i) for i in range(len(self.test_labels[:]))]
        predictions_train = [self.predict(self.train_attributes[:, :], i) for i in range(len(self.train_labels[:]))]
        c = np.sum(predictions_test == self.test_labels[:])
        d = np.sum(predictions_train == self.train_labels[:])
        return c/len(predictions_test), d/len(predictions_train)



computing_times = []

test_accuracies = []
train_accuracies = []
computing_times = []

for i in range(1, 20):
    r = i/10
    r_knn = RadiusKNN(r, train_set, test_set)
    
    tic = time()
    acc_test, acc_train = r_knn.give_accuracy()
    toc = time()
    
    computing_times.append(toc - tic)
    train_accuracies.append(acc_train)
    test_accuracies.append(acc_test)


test_errors = [1 - i for i in test_accuracies]
train_errors = [1 - i for i in train_accuracies]

epoch_range = [i/10 for i in range(1, 20)]

plt.figure(figsize=(12, 8))

plt.title('R Radius Neighbours KNN Train and Test Errors for different r values')
plt.ylabel('Errors')
plt.xlabel('r values')

plt.plot(epoch_range, test_errors, label='Test', color='blue')
plt.plot(epoch_range, train_errors, label='Train', color='red')

plt.xticks(epoch_range)

plt.legend()
plt.show()

