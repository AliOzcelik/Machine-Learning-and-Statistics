# Ali Özçelik 2306579

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from time import time


train_set2 = pd.read_csv("train_set_2.csv").iloc[:, 1:]
test_set2 = pd.read_csv("train_set_2.csv").iloc[:, 1:]

train_set3 = pd.read_csv("train_set_3.csv").iloc[:, 1:]
test_set3 = pd.read_csv("train_set_3.csv").iloc[:, 1:]

train_set4 = pd.read_csv("train_set_4.csv").iloc[:, 1:]
test_set4 = pd.read_csv("train_set_4.csv").iloc[:, 1:]




class NaiveBayes:
    
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.assign_probabilities()
        self.attribute_probs_0, self.attribute_probs_1 = self.assign_attribute_probabilities()
        self.turn_into_array()
        
    def assign_probabilities(self):
        self.class1 = self.train_set[self.train_set['Diagnosis']==1]
        self.class0 = self.train_set[self.train_set['Diagnosis']==0]
        self.c1 = len(self.class1)/len(self.train_set)
        self.c0 = len(self.class0)/len(self.train_set)
        
    def attribute_probabilities(self, attribute_name, class_index):
        if class_index == 1:
            probs = [len(self.class1[self.class1[attribute_name]==i])/len(self.class1) for i in range(4)]
        elif class_index == 0:
            probs = [len(self.class0[self.class0[attribute_name]==i])/len(self.class0) for i in range(4)]
        return probs

    def assign_attribute_probabilities(self):
        columns = self.train_set.columns[:-1]
        attribute_probs_0 = []
        attribute_probs_1 = []
        for i in range(len(columns)):
            attribute_name = columns[i]
            conditional_probs_0 = [self.attribute_probabilities(attribute_name, class_index=0)]
            conditional_probs_1 = [self.attribute_probabilities(attribute_name, class_index=1)]
            attribute_probs_0.append(conditional_probs_0)
            attribute_probs_1.append(conditional_probs_1)
        return attribute_probs_0, attribute_probs_1
    
    def turn_into_array(self):
        self.attribute_probs_0 = np.array(self.attribute_probs_0)
        self.attribute_probs_0 = self.attribute_probs_0.reshape(self.attribute_probs_0.shape[0], self.attribute_probs_0.shape[-1])
        self.attribute_probs_1 = np.array(self.attribute_probs_1)
        self.attribute_probs_1 = self.attribute_probs_1.reshape(self.attribute_probs_1.shape[0], self.attribute_probs_1.shape[-1])
    
    def predict(self, attribute_indexes):
        probability_class_0 = [self.attribute_probs_0[x][int(attribute_indexes[x])] for x in range(10)]
        probability_class_1 = [self.attribute_probs_1[x][int(attribute_indexes[x])] for x in range(10)]
        ali0 = 1
        ali1 = 1
        for i in range(len(probability_class_0)):
            ali0 *= probability_class_0[i]
            ali1 *= probability_class_1[i]
        ali0 *= self.c0
        ali1 *= self.c1
        if ali0 > ali1:
            return 0
        else:
            return 1
        
    
nb2 = NaiveBayes(train_set2, test_set2)
nb3 = NaiveBayes(train_set3, test_set3)
nb4 = NaiveBayes(train_set4, test_set4)

predictions = []

tic = time()
for i in range(len(nb2.test_set)):
    attributes = nb2.test_set.iloc[i, :-1].values
    preds = nb2.predict(attributes)
    predictions.append(preds)
toc = time()
print(toc-tic)


labels = nb2.test_set.iloc[:, -1].values
num_correct = sum(predictions == labels)
accuracy = num_correct/len(labels)
error_rate = 1 - accuracy
print("Accuracy for 2 bins: ", accuracy)
print("Error rate for 2 bins: ", (1-accuracy)*100)



predictions = []

tic = time()
for i in range(len(nb3.test_set)):
    attributes = nb3.test_set.iloc[i, :-1].values
    preds = nb3.predict(attributes)
    predictions.append(preds)
toc = time()
print(toc-tic)


labels = nb3.test_set.iloc[:, -1].values
num_correct = sum(predictions == labels)
accuracy = num_correct/len(labels)
error_rate = 1 - accuracy
print(accuracy)
print("Accuracy for 3 bins: ", accuracy)
print("Error rate for 3 bins: ", (1-accuracy)*100)



predictions = []

tic = time()
for i in range(len(nb4.test_set)):
    attributes = nb4.test_set.iloc[i, :-1].values
    preds = nb4.predict(attributes)
    predictions.append(preds)
toc = time()
print(toc-tic)


labels = nb4.test_set.iloc[:, -1].values
num_correct = sum(predictions == labels)
accuracy = num_correct/len(labels)
error_rate = 1 - accuracy
print(accuracy)

print("Accuracy for 4 bins: ", accuracy)
print("Error rate for 4 bins: ", (1-accuracy)*100)



from sklearn.decomposition import PCA 

dataset = pd.read_excel("Wisconsin Diagnostic Breast Cancer.xlsx")

pca = PCA(n_components=2)

data = pca.fit_transform(dataset.iloc[:, :-1])

labels = dataset.iloc[:, -1].values
labels = labels.reshape(labels.shape[0], 1)
data2 = np.concatenate([data, ], axis=1)

colors = ['red' if label == 0 else 'blue' for label in labels]

plt.title('General Density')
plt.scatter(data2[:, 0], data2[:, 1], c=colors, cmap='Blues')
plt.show()
