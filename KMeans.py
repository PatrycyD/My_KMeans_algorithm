import numpy as np
import pandas as pd

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
            raise AttributeError('Data input needs to be pandas dataframe or numpy array')

    def scale_data(self) -> np.ndarray:

        scaled = np.array([]).reshape(0, self.data.shape[0])

        for column in self.data.T:
            column_mean = np.mean(column)
            column_std = np.std(column)

            scaled_row = list(map(lambda x: (x - column_mean) / column_std, column))
            scaled_row = np.array(scaled_row)

            scaled = np.vstack((scaled, scaled_row))

        return scaled.T

class KMeans():
    def __init__(self, data: np.ndarray, k=5, n_iter=None):
        '''
        KMeas class allows to perform KMeans algorithm on given dataset
        Input: 
        - data - standardized numpy array we want to cluster
        - k - number of clusters
        - n_iter defines number of iterations in algorithm. If none is given algorithm will work until it finished iterating

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

        def calculate_centroids(np_array):
            centroids = np.array([]).reshape(0, np_array.shape[1])
            for k in range(1, self.k + 1):
                current_cluster = np_array[np_array[:, -1] == k]
                centroids = np.vstack((centroids, np.mean(current_cluster, axis=0)))

            return centroids

        def calc_dist_to_clusts(centroids_array, data_array):
            all_distances = np.array([]).reshape(data_array.shape[0], 0)

            for centroid in centroids_array:
                dist_to_all_centr = np.array([]).reshape(0, 1)

                for observation in data_array:
                    dist_to_single_centr = [(observation[i] - centroid[i]) ** 2 for i in range(len(observation))]
                    dist_to_single_centr = np.sqrt(sum(dist_to_single_centr))
                    dist_to_all_centr = np.vstack((dist_to_all_centr, dist_to_single_centr))

                all_distances = np.hstack((all_distances, dist_to_all_centr))

            return all_distances

        def assign_obs_to_clusters(eucl_dists: np.array):
            new_clusters = eucl_dists.argmin(axis=1).reshape(-1, 1) + 1
            self.clustered_data = np.hstack((self.data, new_clusters))

        #assign data to random clusters
        self.random_clusters = np.random.choice(list(range(1, self.k + 1)), len(self.data))
        self.random_clusters = self.random_clusters.reshape(len(self.data), -1)
        self.randomly_clustered = np.hstack((self.data, self.random_clusters))
        self.centroids = calculate_centroids(self.randomly_clustered)

        old_clusters = self.randomly_clustered[:, -1]
        new_clusters = np.array([])

        self.iterations_made = 0
        while ~np.all(old_clusters == new_clusters):
            try:
                old_clusters = self.clustered_data[:, -1]
            except AttributeError:
                old_clusters = old_clusters

            #print('old_clusters', old_clusters)
            #print('new_clusters', new_clusters)
            all_distances = calc_dist_to_clusts(self.centroids, self.data)
            assign_obs_to_clusters(all_distances)

            self.centroids = calculate_centroids(self.clustered_data)
            new_clusters = self.clustered_data[:, -1]
            #print('old_clusters_vol2', old_clusters)
            #print('new_clusters_vol2', new_clusters)
            self.iterations_made += 1

            #print(~np.all(old_clusters == new_clusters))

        # #calulate euclidean distance
        # for iteration in range(self.n_iter):
        #     eucl_dist = np.array([]).reshape(self.data.shape[0],0)

        #     for centroid in centroids:
        #         dist_to_centroid = np.array([]).reshape(0,1)

        #         for d_point in self.data:
        #             distance = np.sqrt(sum([(d_point[i] - centroid[i]) ** 2 for i in range(len(d_point))]))
        #             dist_to_centroid = np.vstack((dist_to_centroid, distance)) 

        #         #stack distances: rows are observations and columns are its distances to every centroid
        #         eucl_dist = np.hstack((eucl_dist, dist_to_centroid))

        #select clusters with shortest distances



        # min_distance = all_distances.argmin(axis=1) + 1
        # clustered_data = np.hstack((self.data, min_distance.reshape(-1, 1)))

        #calculate centroids

        # centroids = np.array([]).reshape(0, self.clustered_data.shape[1])
        # for k in range(1, self.k + 1):
        #     current_cluster = self.clustered_data[self.clustered_data[:,-1] == k]
        #     centroids = np.vstack((centroids, np.mean(current_cluster, axis=0)))
        
        #self.clustered_data = clustered_data
        
        #return clustered_data
    
    def get_df_with_clusters(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        #inputting dataframe into this function will return this dataframe with assigned clusters
        dataframe['cluster'] = self.clustered_data[:,-1]
        dataframe = dataframe.astype({'cluster':int})
        
        return dataframe