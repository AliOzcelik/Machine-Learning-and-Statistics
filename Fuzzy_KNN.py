import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

train_set = pd.read_csv('train_set.csv')
test_set = pd.read_csv('test_set.csv')

class FKNN:
    
    def __init__(self, k, train_set, test_set):
        self.train_set = train_set.iloc[:, 1:]
        self.test_set = test_set.iloc[:, 1:]
        self.k = k
        self.m = 2
        self.split()
        
    def split(self):
        self.train_attributes_set = self.train_set.iloc[:, :-1]
        self.train_labels_set = self.train_set.iloc[:, -1]
        self.test_attributes_set = self.test_set.iloc[:, :-1]
        self.test_labels_set = self.test_set.iloc[:, -1]
        
    def distance(self, vector1, vector2):
        dist = np.linalg.norm(vector1 - vector2)
        return dist
    
    def fuzzification(self, closest_distances, mu):
        ali = 0
        ali2 = 0
        for j in range(len(closest_distances)):
            d = closest_distances[j]
            # in the training set, the closest point is still itself, so distance is zero
            # when turning d = 1/d, it gets zero division error
            # so in the closest distance case i assign it the highest weight
            if d == 0: 
                ali += mu[:, j]
                ali2 += 0.01
            else:
                d = d ** (2 / (self.m - 1))
                d = 1 / d
                ali += mu[:, j] * d
                ali2 += d
        return ali / ali2
    
    # data_point is numpy array 
    # as output, closest k data point indexes are provided in a list
    def closest_neighbours(self, data_point):
        distances = [self.distance(data_point, self.train_attributes_set.iloc[i, :]) for i in range(len(self.train_attributes_set))]
        closest_distances = sorted(distances)[:self.k]
        closest_indexes = [distances.index(i) for i in closest_distances]
        return closest_distances, closest_indexes 
    
    # finds closes neighbours and calcualates their membership values (mu_i based on fuzzy theory)
    def fuzzy_neighbours(self, data_point):
        closest_distances, closest_indexes  = self.closest_neighbours(data_point)
        mu = np.zeros((2, self.k))
        for i in range(len(closest_indexes)):
            if self.train_labels_set.iloc[closest_indexes[i]] == 0:
                mu[0, i] = 1
            else:
                mu[1, i] = 1
        class_membership_value = [self.fuzzification(closest_distances, mu) for x in range(self.k)]
        return class_membership_value
    
    
    def predict(self, data_point):
        predictions = self.fuzzy_neighbours(data_point)
        predictions = np.array(predictions)
        predictions = np.sum(predictions, axis=0)
        if predictions[0] > predictions[1]:
            return 0
        return 1
    
    def give_accuracy(self, data_points, labels):
        correct_predictions = 0
        for i in range(len(data_points)):
            prediction = self.predict(data_points.iloc[i].values)
            if prediction == labels.iloc[i]:
                correct_predictions += 1
        return correct_predictions / len(labels)
    
    def give_accuracy_on_train(self):
        return self.give_accuracy(self.train_attributes_set, self.train_labels_set)
    
    def give_accuracy_on_test(self):
        return self.give_accuracy(self.test_attributes_set, self.test_labels_set)



test_accuracies = []
train_accuracies = []
computing_times = []

for k in range(1, 11):
    fknn = FKNN(k, train_set, test_set)
    
    tic = time()
    acc_train = fknn.give_accuracy_on_train()
    acc_test = fknn.give_accuracy_on_test()
    toc = time()
    computing_times.append(toc - tic)
    train_accuracies.append(acc_train)
    test_accuracies.append(acc_test)


test_errors = [1 - i for i in test_accuracies]
train_errors = [1 - i for i in train_accuracies]

epoch_range = [x+1 for x in range(10)]

plt.title('Fuzzy KNN Train and Test Errors for different k values')
plt.ylabel('Errors')
plt.xlabel('k values')

plt.plot(epoch_range, test_errors, label='Test', color='blue')
plt.plot(epoch_range, train_errors, label='Train', color='red')

plt.xticks(epoch_range)

plt.legend()
plt.show()
    

