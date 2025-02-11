# Ali Özçelik 2306579

import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter('ignore')


# distance used is square of euclidean distance


data = pd.read_excel('dermatology.xlsx')

class KMeansClustering:
    
    def __init__(self, dataset, num_clusters):
        self.k = num_clusters
        self.data = dataset
        self.normalization()
        self.centers = self.initialize_centers()
        self.labels = self.assign(self.centers)
        self.sse = []
        self.temporary_sse = []
        self.data_points = []
        self.computing_time = []
        self.n_iteration = []
        
    
    def normalization(self):
        max_values = np.max(self.data, axis=0)
        min_values = np.min(self.data, axis=0)
        self.data = (self.data - min_values)/(max_values-min_values)
        
    def distance(self, vector1, vector2, axis):
        dist = np.linalg.norm(vector1 - vector2, axis=axis)
        return dist ** 2
    
    def initialize_centers(self):
        covariance_matrix = self.data.cov()
        mean_vector = self.data.mean()
        centers = np.random.multivariate_normal(mean_vector, covariance_matrix, self.k)
        return centers
    
    def assign(self, centers):
        distances = [self.distance(self.data.iloc[x, :].values, centers, 1) for x in range(len(self.data))]
        labels = np.argmin(distances, axis=1)
        return labels
    
    def update_centers(self):
        new_centers = [np.array(self.data[self.labels == x]) for x in range(self.k)]
        new_centers = np.array([x.mean(axis=0) if len(x) > 0 else np.zeros(x.shape[1]) for x in new_centers])
        return new_centers
    
    def number_of_data_points(self):
        data_points = [np.array(self.data[self.labels == x]) for x in range(self.k)]
        data_points = [x.shape[0] for x in data_points]
        return data_points
    
    def calculate_sse(self):
        local_sse = []
        data_points = [(np.array(self.data[self.labels == x]) )for x in range(self.k)]
        for i in range(self.k):
            local_sse_2 = 0
            if len(self.centers[i]) > 0:
                for j in data_points[i]:
                    local_sse_2 += self.distance(j, self.centers[i], 0)
                local_sse.append(local_sse_2)
        self.temporary_sse.append(local_sse)
        
        
    
    def iterate(self, num_iterations):
        tic = time.time()
        n_iteration = 0
        for i in range(num_iterations):    
            new_centers = self.update_centers()
            new_labels = self.assign(new_centers)
            self.calculate_sse()
            
            if (new_labels == self.labels).all():
                self.centers = new_centers
                n_iteration = i
                break
            
            else:
                 self.labels == new_labels
            
            if (new_centers == self.centers).all():
                n_iteration = i
                break
            
            else:
                self.centers = new_centers
                self.labels = new_labels
                
            
        toc = time.time()
        self.computing_time.append(toc-tic)
        self.n_iteration = n_iteration
        self.sse = sum(self.temporary_sse[-1])
        self.data_points = self.number_of_data_points()



total_sse = []
total_data_points = []
total_n_iterations = []
total_computing_time = []
for i in range(500):
    k_means = KMeansClustering(data, num_clusters=10) 
    k_means.iterate(500)
    total_sse.append(k_means.sse)
    total_data_points.append(k_means.data_points)
    total_n_iterations.append(k_means.n_iteration)
    total_computing_time.extend(k_means.computing_time)
    


# =============================================================================
# In the RESULTS Part
# 
# min(total_sse)
# 362.7060809481986
# 
# total_sse.index(min(total_sse))
# 87
# 
# total_data_points[87]
# [21, 46, 52, 20, 66, 32, 43, 13, 43, 29]
# 
# sum(total_computing_time)
# 38.35259938240051
# =============================================================================
