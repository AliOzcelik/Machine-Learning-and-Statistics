# Ali Özçelik 2306579

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

dataset = pd.read_excel('/Users/desidero/Desktop/Dersler/IE460/assignment 2/Wisconsin Diagnostic Breast Cancer.xlsx')
columns = dataset.columns

#num_bins = 4

#len_bins = [math.floor(len(dataset)/num_bins) for x in range(num_bins-1)]
#len_bins.append(len(dataset)-sum(len_bins))


#num_bins = len(len_bins)
#medians = [np.median(dataset.iloc[:, i].values) for i in range(10)]


num_bins = 2
cols2 = []
if num_bins == 2:
    for i in range(10):
        attribute = []
        arr = dataset.iloc[:, i].values
        median = np.median(arr)
        for j in range(len(arr)):
            if arr[j] <= median:
                attribute.append(0)
            else:
                attribute.append(1)
        cols2.append(attribute)

col2 = np.transpose(np.array(cols2))

fig, axs = plt.subplots(2, 5, figsize=(25, 10))
for i in range(11):
    axs[i // 5, i % 5].hist(col2[:, i], bins=4)
    axs[i // 5, i % 5].tick_params(axis='both', which='major', length=4)
    axs[i // 5, i % 5].set_xlabel(f'{dataset.columns[i].capitalize()}', fontsize=12)
    axs[i // 5, i % 5].set_ylabel('Frequency', fontsize=12)

plt.show()




num_bins = 3
cols3 = []
if num_bins == 3:
    for i in range(10):
        attribute = []
        arr = dataset.iloc[:, i].values
        imaginary = sorted(list(arr))
        divide1 = imaginary[math.ceil(len(imaginary)/3)]
        divide2 = imaginary[math.ceil(2*len(imaginary)/3)]
        for j in range(len(arr)):
            if arr[j] <= divide1:
                attribute.append(0)
            elif arr[j] <= divide2:
                attribute.append(1)
            else:
                attribute.append(2)
        cols3.append(attribute)

col3 = np.transpose(np.array(cols3))

fig, axs = plt.subplots(2, 5, figsize=(25, 10))
for i in range(11):
    axs[i // 5, i % 5].hist(col3[:, i], bins=4)
    axs[i // 5, i % 5].tick_params(axis='both', which='major', length=4)
    axs[i // 5, i % 5].set_xlabel(f'{dataset.columns[i].capitalize()}', fontsize=12)
    axs[i // 5, i % 5].set_ylabel('Frequency', fontsize=12)

plt.show()


num_bins = 4
cols4 = []
if num_bins == 4:
    for i in range(10):
        attribute = []
        arr = dataset.iloc[:, i].values
        median = np.median(arr)
        min_arr = np.min(arr)
        max_arr = np.max(arr)
        arr_min = arr[arr <= median]
        arr_max = arr[arr > median]
        median1 = np.median(arr_min)
        #arr_min_lower = arr_min[arr_min <= median]
        #arr_min_upper = arr_min[arr_min > median]
        median2 = np.median(arr_max)
        #arr_max_lower = arr_max[arr_max <= median2]
        #arr_max_upper = arr_max[arr_max > median2]
        for j in range(len(arr)):
            if arr[j] <= median1:
                attribute.append(0)
            elif arr[j] <= median:
                attribute.append(1)
            elif arr[j] <= median2:
                attribute.append(2)
            else:
                attribute.append(3)
        cols4.append(attribute)

col4 = np.transpose(np.array(cols4))

fig, axs = plt.subplots(2, 5, figsize=(25, 10))
for i in range(11):
    axs[i // 5, i % 5].hist(col4[:, i], bins=4)
    axs[i // 5, i % 5].tick_params(axis='both', which='major', length=4)
    axs[i // 5, i % 5].set_xlabel(f'{dataset.columns[i].capitalize()}', fontsize=12)
    axs[i // 5, i % 5].set_ylabel('Frequency', fontsize=12)

plt.show()

labels = dataset.iloc[:, -1].values.reshape([dataset.iloc[:, -1].values.shape[0], 1])

col2_2 = np.concatenate((col2, labels),axis=1)
dataset2 = pd.DataFrame(col2_2, columns=columns)

col3_2 = np.concatenate((col3, labels),axis=1)
dataset3 = pd.DataFrame(col3_2, columns=columns)

col4_2 = np.concatenate((col4, labels),axis=1)
dataset4 = pd.DataFrame(col4_2, columns=columns)


def shuffle(data):
    columns = data.columns
    ali = np.zeros((data.iloc[:, :].values.shape))
    shuffled_list = random.sample(range(len(data)), len(data))
    for i in range(len(data)):
        ali[i, :] = data.iloc[shuffled_list[i], :]
    ali = pd.DataFrame(ali, columns=columns)
    return ali    

shuffled_data2 = shuffle(dataset2)
shuffled_data3 = shuffle(dataset3)
shuffled_data4 = shuffle(dataset4)

def form_train_test(dataset, index=0.8):
    divide = math.floor(len(dataset)*index)
    train_set = dataset.iloc[:divide, :]
    test_set = dataset.iloc[divide:, :]
    return train_set, test_set

train_set2, test_set2 = form_train_test(shuffled_data2, index=0.8)
train_set3, test_set3 = form_train_test(shuffled_data3, index=0.8)
train_set4, test_set4 = form_train_test(shuffled_data4, index=0.8)

train_set2.to_csv("/Users/desidero/Desktop/Dersler/IE460/assignment 4/CA4_ID2306579_Part1A/train_set_2.csv")
test_set2.to_csv("/Users/desidero/Desktop/Dersler/IE460/assignment 4/CA4_ID2306579_Part1A/test_set_2.csv")

train_set3.to_csv("/Users/desidero/Desktop/Dersler/IE460/assignment 4/CA4_ID2306579_Part1A/train_set_3.csv")
test_set3.to_csv("/Users/desidero/Desktop/Dersler/IE460/assignment 4/CA4_ID2306579_Part1A/test_set_3.csv")

train_set4.to_csv("/Users/desidero/Desktop/Dersler/IE460/assignment 4/CA4_ID2306579_Part1A/train_set_4.csv")
test_set4.to_csv("/Users/desidero/Desktop/Dersler/IE460/assignment 4/CA4_ID2306579_Part1A/test_set_4.csv")

