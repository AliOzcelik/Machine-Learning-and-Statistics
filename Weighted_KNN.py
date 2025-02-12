import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from time import time

dataset = pd.read_excel('Wisconsin Diagnostic Breast Cancer.xlsx')


def shuffle(data):
    columns = data.columns
    ali = np.zeros((data.iloc[:, :].values.shape))
    shuffled_list = random.sample(range(len(data)), len(data))
    for i in range(len(data)):
        ali[i, :] = data.iloc[shuffled_list[i], :]
    ali = pd.DataFrame(ali, columns=columns)
    return ali    


def normalize(dataset):
    max_values = np.max(dataset, axis=0)
    min_values = np.min(dataset, axis=0)
    dataset = (dataset - min_values)/(max_values-min_values)
    return dataset


dataset = normalize(dataset)
dataset = shuffle(dataset)

def train_test_split(dataset, index):
    split_index = int(len(dataset)*index)
    train_set = dataset.iloc[:split_index, :]
    test_set = dataset.iloc[split_index:, :]
    return train_set, test_set

train_set, test_set = train_test_split(dataset, 0.8)

class ModifiedKNN:
    
    def __init__(self, k, train_set, test_set):
        self.k = k
        self.train_set = train_set.iloc[:, :].values
        self.test_set = test_set.iloc[:, :].values
        self.split()
    
    def split(self):
        self.train_attributes = self.train_set[:, :-1]
        self.train_labels = self.train_set[:, -1]
        self.test_attributes = self.test_set[:, :-1]
        self.test_labels = self.test_set[:, -1]
    
    def distance(self, vector1, vector2):
        dist = np.linalg.norm(vector1 - vector2)
        return dist 

    def assign_weights(self, distances):
        max_distance = max(distances)
        min_distance = min(distances)
        weights = [(max_distance - x)/(max_distance -  min_distance) for x in distances]
        return weights
    
    def predict(self, dataset, data_point_index): 
        data_point = dataset[data_point_index, :]
        distances = [self.distance(data_point, self.train_attributes[i, :]) for i in range(len(self.train_attributes))]
        weights = self.assign_weights(distances)
        closest_distances = sorted(distances)[:self.k]
        closest_indexes = [distances.index(i) for i in closest_distances]
        closest_labels = [self.train_labels[i] for i in closest_indexes]
        class_weights = np.zeros((len(set(dataset[:, -1]))))
        for i in range(k):
            class_weights[int(closest_labels[i])] += weights[closest_indexes[i]]
            
        if class_weights[0] >  class_weights[1]:
            return 0
        else:
            return 1
    
    
    def give_accuracy(self): 
        predictions_test = [self.predict(self.test_attributes, i) for i in range(len(self.test_labels))]
        predictions_train = [self.predict(self.train_attributes, i) for i in range(len(self.train_labels))]
        c = np.sum(predictions_test == self.test_labels[:])
        d = np.sum(predictions_train == self.train_labels[:])
        return c/len(predictions_test), d/len(predictions_train), predictions_test
         



computing_times = []

test_accuracies = []
train_accuracies = []

predictions_my = []

wknn_accuracy_minor = []

for k in range(1,21):
    
    tic = time()
    
    modified_knn = ModifiedKNN(k, train_set, test_set)
    test_accuracy, train_accuracy, predictions_test = modified_knn.give_accuracy()
    
    toc = time()
    
    computing_times.append(toc-tic)
        
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)
    
    
