# Ali Özçelik 2306579

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from time import time


data = pd.read_excel('/Users/desidero/Desktop/Dersler/IE460/assignment 2/Wisconsin Diagnostic Breast Cancer.xlsx')
    
def normalize(dataset):
    max_values = np.max(dataset, axis=0)
    min_values = np.min(dataset, axis=0)
    dataset = (dataset - min_values)/(max_values-min_values)
    return dataset

data2 = normalize(data)


# dataset is pandas dataframe
# train_size if fraction 
def random_partition(dataset, train_size=0.8):
    train_amount = int(len(dataset)*train_size)
    shuffled_list = np.random.permutation(len(dataset))
    train_set = dataset.iloc[shuffled_list[:train_amount], :]
    test_set = dataset.iloc[shuffled_list[train_amount:], :]
    return train_set, test_set

train_set, test_set = random_partition(data2)


def split(dataset):
    attributes = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    return attributes, labels

train_attributes, train_labels = split(train_set)
test_attributes, test_labels = split(test_set)

def save():
    train_attributes.to_excel('/Users/desidero/Desktop/Dersler/IE460/assignment 2/CA_ID2306579_PartA/train_attributes.xlsx')
    test_attributes.to_excel('/Users/desidero/Desktop/Dersler/IE460/assignment 2/CA_ID2306579_PartA/test_attributes.xlsx')
    train_labels.to_excel('/Users/desidero/Desktop/Dersler/IE460/assignment 2/CA_ID2306579_PartA/train_labels.xlsx')
    test_labels.to_excel('/Users/desidero/Desktop/Dersler/IE460/assignment 2/CA_ID2306579_PartA/test_labels.xlsx')



class KNN:
    
    def __init__(self, k, train_attributes_set, train_labels_set, test_attributes_set, test_labels_set):
        self.train_attributes_set = train_attributes_set #.iloc[:, 1:]
        self.train_labels_set = train_labels_set #.iloc[:, 1:]
        self.test_attributes_set = test_attributes_set #.iloc[:, 1:]
        self.test_labels_set = test_labels_set #.iloc[:, 1:]
        self.k = k
        self.test_accuracy, self.train_accuracy = self.give_accuracy()
        
    def distance(self, vector1, vector2):
        dist = np.linalg.norm(vector1 - vector2)
        return dist ** 2
    
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
        distances = [self.distance(data_point, self.train_attributes_set.iloc[i, :]) for i in range(len(self.train_attributes_set))]
        closest_distances = sorted(distances)[:self.k]
        closest_indexes = [distances.index(i) for i in closest_distances]
        return closest_indexes 

    # labels are numpy array
    # dataset is numpy array
    def predict(self, dataset, data_point_index):
        data_point = dataset[data_point_index, :]
        closest_indexes = self.closest_neighbours(data_point, data_point_index)
        closest_labels = [self.train_labels_set.iloc[i] for i in closest_indexes]
        
        if self.count(closest_labels):
            return 1
        return 0


    # labels are numpy array
    def give_accuracy(self):
        predictions_test = [self.predict(self.test_attributes_set.iloc[:, :].values, i) for i in range(len(self.test_labels_set.iloc[:].values))]
        predictions_train = [self.predict(self.train_attributes_set.iloc[:, :].values, i) for i in range(len(self.train_labels_set.iloc[:].values))]
        c = np.sum(predictions_test == self.test_labels_set.iloc[:].values)
        d = np.sum(predictions_train == self.train_labels_set.iloc[:].values)
        return c/len(predictions_test), d/len(predictions_train)
        


def plot(x, y, train_or_test):
    plt.xlabel('Number of Nearest Neighbuors')
    plt.ylabel('Accuracy')
    plt.title('{} set accuracy with different k values'.format(train_or_test))
    plt.plot(x, y)
    plt.xticks([x + 1 for x in range(20)])
    plt.show()

def plot2(x, y, train_or_test):
    plt.xlabel('Number of Nearest Neighbuors')
    plt.ylabel('Errors')
    plt.title('{} set error with different k values'.format(train_or_test))
    plt.plot(x, y)
    plt.xticks([x + 1 for x in range(20)])
    plt.show()
    


test_accuracies = []
train_accuracies = []
computing_times = []

for k in range(1, 21):
    tic = time()
    knn = KNN(k, train_attributes, train_labels, test_attributes, test_labels)
    toc = time()
    computing_times.append(toc-tic)
    test_accuracies.append(knn.test_accuracy)
    train_accuracies.append(knn.train_accuracy)
    
errors_test = [1-i for i in test_accuracies]
errors_train = [1-i for i in train_accuracies]

plot([x for x in range(1, 21)], test_accuracies, 'test')
plot2([x for x in range(1, 21)], errors_test, 'test')

plot([x for x in range(1, 21)], train_accuracies, 'train')
plot2([x for x in range(1, 21)], errors_train, 'train')



