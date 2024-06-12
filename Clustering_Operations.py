import numpy as np
import cv2
import skimage as ski
from Data import Point_Matrix

from sklearn import cluster


class Clustering_Operations( Point_Matrix ):
    """
    @brief Clustering operations class
    @details This class is responsible for clustering operations.
    """
    def __init__(self, data: Point_Matrix):
        """
        @brief Constructor
        @param data: Point_Matrix object
        """
        super().__init__(data = data)

    # Method handler for clustering
    def method_handler_clustering(self,method_name, args_dict):
        """
        @brief Method handler for clustering
        @param method_name: str
        @param args_dict: dict
        """

        if method_name == 'K-Means':
            self.kmeans(
                n_clusters = args_dict['n_clusters'], max_iter = args_dict['max_iter'], init = args_dict['init'], algorithm = args_dict['algorithm']
                )
        elif method_name == 'Affinity Propagation':
            self.affinity_propagation(
                damping = args_dict['damping'], max_iter = args_dict['max_iter'], convergence_iter = args_dict['convergence_iter']
                )
        elif method_name == 'Mean Shift':
            self.mean_shift(
                bandwidth = args_dict['bandwidth'], max_iter = args_dict['max_iter']
                )
        elif method_name == 'Spectral Clustering':
            self.spectral_clustering(
                n_clusters = args_dict['n_clusters'], assign_labels = args_dict['assign_labels'],
                eigen_solver = args_dict['eigen_solver'], random_state = args_dict['random_state']
                )
        elif method_name == 'Hierarchical Clustering':
            self.hierarchical_clustering(
                n_clusters = args_dict['n_clusters'], linkage = args_dict['linkage'], distance_threshold = args_dict['distance_threshold']
                )
        elif method_name == 'DBSCAN':
            return self.dbscan(
                eps = args_dict['eps'], min_samples = args_dict['min_samples'], metric = args_dict['metric']
                )
        else:
            print("Method not found.")


    ####################################### CLUSTERING METHODS #######################################

    def kmeans(self, n_clusters = 3, max_iter = 300, init = 'k-means++', algorithm = 'auto'):
        """
        @brief KMeans clustering
        @param n_clusters: int
        @param max_iter: int
        @param init: str
        @param algorithm: str
        """
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

        self.calculate_center_nodes()

    def affinity_propagation(self, damping = 0.5, max_iter = 200, convergence_iter = 15):
        """
        @brief Affinity Propagation clustering
        @param damping: float
        @param max_iter: int
        @param convergence_iter: int
        """

        # Get data
        data = np.array(self.get_data_as_list())
        
        # Affinity Propagation
        affinity_propagation = cluster.AffinityPropagation(damping = damping, max_iter = max_iter, convergence_iter = convergence_iter)
        affinity_propagation.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(affinity_propagation.labels_)
        
    def mean_shift(self, bandwidth = 250, max_iter = 300):
        """
        @brief Mean Shift clustering
        @param bandwidth: float
        @param max_iter: int
        """

        # Get data
        data = np.array(self.get_data_as_list())
        
        # Mean Shift
        mean_shift = cluster.MeanShift(bandwidth = bandwidth, max_iter = max_iter)
        mean_shift.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(mean_shift.labels_)

    def spectral_clustering(self, n_clusters = 8, assign_labels = 'kmeans', eigen_solver = None, random_state = None):
        """
        @brief Spectral Clustering
        @param n_clusters: int
        @param assign_labels: str
        @param eigen_solver: str
        @param random_state: int
        """
        
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Spectral Clustering
        spectral_clustering = cluster.SpectralClustering(n_clusters = n_clusters, assign_labels = assign_labels, eigen_solver = eigen_solver, random_state = random_state)
        spectral_clustering.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(spectral_clustering.labels_)

    def hierarchical_clustering(self, n_clusters = 2, linkage = 'ward', distance_threshold = None):
        """
        @brief Hierarchical Clustering
        @param n_clusters: int
        @param linkage: str
        @param distance_threshold: float
        """

        # Get data
        data = np.array(self.get_data_as_list())
        
        # Hierarchical Clustering
        hierarchical_clustering = cluster.AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage, distance_threshold = distance_threshold)
        hierarchical_clustering.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(hierarchical_clustering.labels_)

    def dbscan(self, eps = 0.5, min_samples = 5, metric = 'euclidean'):
        """
        @brief DBSCAN clustering
        @param eps: float
        @param min_samples: int
        @param metric: str
        """

        # Get data
        data = np.array(self.get_data_as_list())
        
        # DBSCAN
        dbscan = cluster.DBSCAN(eps = eps, min_samples = min_samples, metric = metric)
        dbscan.fit(data)

        # If DBSCAN could not find any cluster, try different parameters
        if len(dbscan.labels_) == 0 or -1 in dbscan.labels_:
            target_class = 3

            print("DBSCAN could not find any cluster properly. Trying different parameters...( n clusters = {} )".format(target_class+1))
            
            eps_list = np.linspace(data.min(), data.max(), 100)
            min_samples_list =np.linspace(1, 100, 100)

            for eps in eps_list:
                for min_samples in min_samples_list:
                    dbscan = cluster.DBSCAN(eps = eps, min_samples = int(min_samples), metric = metric)
                    dbscan.fit(data)
                    if max(dbscan.labels_) == target_class and -1 not in dbscan.labels_:          
                        self.set_cluster_vector(dbscan.labels_)
                        return {"eps": eps, "min_samples": min_samples, "metric": metric, "auto": True}
        else:
            self.set_cluster_vector(dbscan.labels_)
            return {"eps": eps, "min_samples": min_samples, "metric": metric, "auto": False}
