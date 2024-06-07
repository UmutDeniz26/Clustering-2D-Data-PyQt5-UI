import numpy as np
import cv2
import skimage as ski
from Data import Point_Matrix

from sklearn import cluster


class Clustering_Operations( Point_Matrix ):
    def __init__(self, data: Point_Matrix):
        """
        Constructor for image_operator class
        :param image: np.ndarray or str
        """
        Point_Matrix.__init__(self, data)
        
    def method_handler_clustering(self,method_name, args_dict):
        if method_name == 'K-Means':
            self.kmeans(n_clusters = args_dict['n_clusters'], max_iter = args_dict['max_iter'], init = args_dict['init'], algorithm = args_dict['algorithm'])
        elif method_name == 'Affinity Propagation':
            self.affinity_propagation()
        elif method_name == 'Mean Shift':
            self.mean_shift()
        elif method_name == 'Spectral Clustering':
            self.spectral_clustering()
        elif method_name == 'Hierarchical Clustering':
            self.hierarchical_clustering()
        elif method_name == 'DBSCAN':
            self.dbscan()
        else:
            print("Method not found.")

    def kmeans(self, n_clusters = 3, max_iter = 300, init = 'k-means++', algorithm = 'auto'):
        # Get data
        
        data = np.array(self.get_data_as_list())
        
        # KMeans
        if algorithm == 'auto':
            kmeans = cluster.KMeans(n_clusters = n_clusters, max_iter = max_iter, init = init)
        else:
            kmeans = cluster.KMeans(n_clusters = n_clusters, max_iter = max_iter, init = init, algorithm = algorithm)
        kmeans.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(kmeans.labels_)
        self.set_cluster_id_vector(kmeans.cluster_centers_)
        self.set_result(kmeans)


    def affinity_propagation(self):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Affinity Propagation
        affinity_propagation = cluster.AffinityPropagation()
        affinity_propagation.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(affinity_propagation.labels_)
        self.set_cluster_id_vector(affinity_propagation.cluster_centers_indices_)
        self.set_result(affinity_propagation)
        
    def mean_shift(self):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Mean Shift
        mean_shift = cluster.MeanShift()
        mean_shift.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(mean_shift.labels_)
        self.set_cluster_id_vector(mean_shift.cluster_centers_)
        self.set_result(mean_shift)

    def spectral_clustering(self):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Spectral Clustering
        spectral_clustering = cluster.SpectralClustering()
        spectral_clustering.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(spectral_clustering.labels_)
        self.set_result(spectral_clustering)
        
    def hierarchical_clustering(self):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Hierarchical Clustering
        hierarchical_clustering = cluster.AgglomerativeClustering()
        hierarchical_clustering.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(hierarchical_clustering.labels_)
        self.set_cluster_id_vector(hierarchical_clustering.children_)
        self.set_result(hierarchical_clustering)


    def dbscan(self):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # DBSCAN
        dbscan = cluster.DBSCAN()
        dbscan.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(dbscan.labels_)
        self.set_cluster_id_vector(dbscan.components_)

if __name__ == '__main__':
    pass
