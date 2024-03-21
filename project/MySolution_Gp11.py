import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from utils import *
### TODO: import any other packages you need for your solution

#--- Task 1 ---#

class MyClassifier:  
    def __init__(self, K):
        self.K = K  # Number of classes
        self.W = None
        self.b = None
        self.label_map = None

    def _one_hot_encode(self, labels):
        unique_labels = np.unique(labels)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        one_hot = np.zeros((len(labels), self.K))
        for idx, label in enumerate(labels):
            one_hot[idx, self.label_map[label]] = 1
        return one_hot

    def train(self, trainX, trainY):
        N = trainX.shape[1]  # Number of features
        M = trainX.shape[0]  # Number of training samples

        y_train = self._one_hot_encode(trainY)  # One-hot encode trainY

        W = cp.Variable((self.K, N))
        b = cp.Variable(self.K)
        slack = cp.Variable((M, self.K))

        # Objective: Minimize the sum of slack variables (change to linear SVM)
        objective = cp.Minimize(cp.sum(slack))

        # Constraints for hinge loss
        constraints = []
        for i in range(M):
            for k in range(self.K):
                y_binary = 1 if y_train[i, k] == 1 else -1
                constraints.append(y_binary * (W[k] @ trainX[i] + b[k]) >= 1 - slack[i, k])
                constraints.append(slack[i, k] >= 0)  # Slack variables must be non-negative

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)

        self.W = W.value
        self.b = b.value

    def predict(self, testX):
        raw_pred = testX @ self.W.T + self.b
        pred_idx = np.argmax(raw_pred, axis=1)
        predY = np.array([list(self.label_map.keys())[list(self.label_map.values()).index(i)] for i in pred_idx])
        return predY

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)
        return accuracy
    def predict_raw_scores(self, X):
        return X @ self.W.T + self.b


##########################################################################
#--- Task 2 ---#
class MyClustering:
    
    def __init__(self, K):
        
        self.K = K  # number of classes
        self.labels = None
        self.cluster_centers_ = None
        
        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
    def train(self, trainX):
        N, d = trainX.shape
        tolerance = 1e-3  # Set the tolerance value

        # Initialize cluster centers by randomly choosing K data points from trainX
        random_indices = np.random.choice(N, self.K, replace=False)
        self.cluster_centers_ = trainX[random_indices, :]

        iteration = 0
        while iteration < 100:
            # Flatten cluster centers to use in the optimization
            flat_centers = self.cluster_centers_.flatten()

            # Linear programming formulation for clustering
            c = np.ones(N * self.K)
            A_eq = np.zeros((N, N * self.K))
            for i in range(N):
                A_eq[i, i * self.K: (i + 1) * self.K] = 1
            b_eq = np.ones(N)
            bounds = [(0, 1) for _ in range(N * self.K)]

            # Solve the linear programming problem
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            # print(result.x.shape)
            new_labels = np.argmax(result.x.reshape(N, self.K), axis=1)

            # Check for convergence based on tolerance
            if np.array_equal(new_labels, self.labels):
                print("Converged.",iteration)
                break

            # Update cluster centers based on the new assignments
            self.labels = new_labels
            self.cluster_centers_ = np.array([np.mean(trainX[self.labels == k], axis=0) for k in range(self.K)])

            iteration += 1

        return self.labels


    def infer_data_labels(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
            
        '''
        if self.cluster_centers_ is None:
            raise ValueError("Model not trained. Please call train() first.")

        # Assign each data point to the nearest cluster
        pred_labels = np.argmin(np.linalg.norm(testX[:, np.newaxis, :] - self.cluster_centers_, axis=2), axis=1)

        

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_data_labels(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables


##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio, num_classes):
        self.ratio = ratio  # percentage of data to label
        self.num_classes = num_classes  # number of clusters


    def select(self, trainX):
        X_normalized = normalize(trainX, axis=0)#
        kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(X_normalized)
        labels = kmeans.labels_

        num_samples_to_label = int(trainX.shape[0] * self.ratio)
        select_point = cp.Variable(trainX.shape[0], boolean=True)
        select_cluster = cp.Variable(self.num_classes, boolean=True)

        # Objective - Maximizing the number of clusters represented
        objective = cp.Maximize(cp.sum(select_cluster))

        # Constraints
        constraints = [cp.sum(select_point) == num_samples_to_label]
        for k in range(self.num_classes):
            cluster_indices = (labels == k)
            constraints.append(select_cluster[k] <= cp.sum(select_point[cluster_indices]))

        problem = cp.Problem(objective, constraints)
        problem.solve()#solver=cp.GLPK_MI)

        selected_indices = np.where(select_point.value >= 0.5)[0]
        data_to_label = trainX[selected_indices, :]
        return data_to_label
    
    
    def randomly_select(self, trainX):
        if self.ratio == 1:
            return trainX
        num_samples_to_label = int(trainX.shape[0] * self.ratio)
        # Randomly choose indices without replacement
        selected_indices = np.random.choice(trainX.shape[0], num_samples_to_label, replace=False)
        data_ramdomly_label = trainX[selected_indices, :]
        return data_ramdomly_label
    
 
