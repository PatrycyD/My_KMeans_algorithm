import numpy as np
import pandas as pd

df = pd.read_csv('iris_dataset.csv')

df.drop('species', axis=1, inplace=True)

class Scaler:
    '''
    Takes pandas dataframe or numpy array as input.
    Its only method standardize data and return numpy array
    '''
    def __init__(self, dataframe: pd.DataFrame):
        if isinstance(dataframe, pd.core.frame.DataFrame):
            self.data = dataframe.to_numpy()
        elif isinstance(dataframe, np.ndarray):
            self.data = dataframe
        else:
            raise TypeError('Data input need to be pandas dataframe or numpy array')

    def scale_data(self) -> np.ndarray:
        
        scaled = np.array([]).reshape(0, self.data.shape[0])

        for column in self.data.T:
            column_mean = np.mean(column)
            column_std = np.std(column)

            scaled_row = list(map(lambda x: (x - column_mean)/column_std, column))
            scaled_row = np.array(scaled_row)

            scaled = np.vstack((scaled, scaled_row))

        return scaled.T
    

class KMeans():
    def __init__(self, data: np.ndarray, k=5, n_iter=10):
        '''
        KMeas class allows to perform KMeans algorithm on given dataset
        Input: 
        - data - standardized numpy array we want to cluster
        - k - number of clusters
        - n_iter defines number of iterations in algorithm

        Its only method perform whole algorithm at once: 
        As a first step assigns every observation to random clusters, then calculate cetroids fot these clusters.
        Later assigns observations to closest centroids. Repeats second and third step n_iter times 
        '''
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError('Data input need to be numpy array')

        self.k = k
        self.n_iter = n_iter
        
    def get_clusters(self):
        #assign data to random clusters
        random_clusters = np.random.choice(list(range(1, self.k + 1)), len(self.data))
        random_clusters = random_clusters.reshape(len(self.data), -1)
        rand_clust = np.hstack((self.data, random_clusters))

        #calculate centroids for random clusters
        centroids = np.array([]).reshape(0, rand_clust.shape[1])
        for k in range(1, self.k + 1):
            current_cluster = rand_clust[rand_clust[:,-1] == k]
            centroids = np.vstack((centroids, np.mean(current_cluster, axis=0)))
        
        #calulate euclidean distance
        for iteration in range(self.n_iter):
            
            eucl_dist = np.array([]).reshape(self.data.shape[0],0)
            for centroid in centroids:
                dist_to_centroid = np.array([]).reshape(0,1)
                
                for d_point in self.data:
                    distance = np.sqrt(sum([(d_point[i] - centroid[i]) ** 2 for i in range(len(d_point))]))
                    dist_to_centroid = np.vstack((dist_to_centroid, distance)) 
                
                #stack distances: rows are observations and columns are its distances to every centroid
                eucl_dist = np.hstack((eucl_dist, dist_to_centroid))
                
            #select clusters with shortest distances
            min_distance = eucl_dist.argmin(axis=1) + 1
            clustered_data = np.hstack((self.data, min_distance.reshape(-1, 1)))
            
            #calculate centroids
            centroids = np.array([]).reshape(0, clustered_data.shape[1])
            for k in range(1, self.k + 1):
                current_cluster = clustered_data[clustered_data[:,-1] == k]
                centroids = np.vstack((centroids, np.mean(current_cluster, axis=0)))
            
        self.centroids = centroids
        self.clustered_data = clustered_data
        
        return clustered_data
    
    def get_df_with_clusters(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        #inputting dataframe into this function will return this dataframe with assigned clusters
        dataframe['cluster'] = self.clustered_data[:,-1]
        dataframe = dataframe.astype({'cluster':int})
        
        return dataframe