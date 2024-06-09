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
        Point_Matrix.__init__(self, data = data)

    # Method handler for clustering
    def method_handler_clustering(self,method_name, args_dict):
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
            self.dbscan(
                eps = args_dict['eps'], min_samples = args_dict['min_samples'], metric = args_dict['metric']
                )
        else:
            print("Method not found.")

    def get_cluster_points(self):
        # Get cluster points
        count_clusters = max(self.get_cluster_vector()) + 1 

        self.cluster_points = [[] for i in range(count_clusters)]

        for i in range(count_clusters):
            for point in self.get_data():
                if point.get_cluster_id() == i:
                    self.cluster_points[i].append(point.get_coordinates())
                    
        return self.cluster_points

    def get_center_nodes(self):
        
        # Minimum distance between a point and a cluster, will be the center of the cluster
        count_clusters = max(self.get_cluster_vector()) + 1
        self.center_nodes = []
        for i in range(count_clusters):
            min_distance = float('inf')
            center_node = None
            for point in self.get_data():
                if point.get_cluster_id() == i:
                    distance = np.linalg.norm(point.get_coordinates() - self.calculate_cluster_centers()[i])
                    if distance < min_distance:
                        min_distance = distance
                        center_node = point.get_coordinates()
            self.center_nodes.append(center_node)
        
        return self.center_nodes

    def get_cluster_items(self):
        # Get cluster items
        count_clusters = max(self.get_cluster_vector()) + 1

        cluster_items = {i: [] for i in range(count_clusters)}
        points = self.get_data()

        for i in range(count_clusters):
            for point in points:
                if point.get_cluster_id() == i:
                    cluster_items[i].append(point)
                    
        return cluster_items
    

    # Function to calculate cluster centers for other methods
    def calculate_cluster_centers(self):
        unique_labels = np.unique(self.get_cluster_vector())
        data_points = np.array(self.get_data_as_list())
        centers = np.array([data_points[self.get_cluster_vector() == label].mean(axis=0) for label in unique_labels])
        return centers




    ####################################### CLUSTERING METHODS #######################################

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
        self.set_result(kmeans)

        self.get_center_nodes()

    def affinity_propagation(self, damping = 0.5, max_iter = 200, convergence_iter = 15):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Affinity Propagation
        affinity_propagation = cluster.AffinityPropagation(damping = damping, max_iter = max_iter, convergence_iter = convergence_iter)
        affinity_propagation.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(affinity_propagation.labels_)
        self.set_result(affinity_propagation)
        
    def mean_shift(self, bandwidth = None, max_iter = 300):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Mean Shift
        mean_shift = cluster.MeanShift(bandwidth = bandwidth, max_iter = max_iter)
        mean_shift.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(mean_shift.labels_)
        self.set_result(mean_shift)

    def spectral_clustering(self, n_clusters = 8, assign_labels = 'kmeans', eigen_solver = None, random_state = None):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Spectral Clustering
        spectral_clustering = cluster.SpectralClustering(n_clusters = n_clusters, assign_labels = assign_labels, eigen_solver = eigen_solver, random_state = random_state)
        spectral_clustering.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(spectral_clustering.labels_)
        self.set_result(spectral_clustering)

    def hierarchical_clustering(self, n_clusters = 2, linkage = 'ward', distance_threshold = None):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # Hierarchical Clustering
        hierarchical_clustering = cluster.AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage, distance_threshold = distance_threshold)
        hierarchical_clustering.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(hierarchical_clustering.labels_)
        self.set_result(hierarchical_clustering)

    def dbscan(self, eps = 0.5, min_samples = 5, metric = 'euclidean'):
        # Get data
        data = np.array(self.get_data_as_list())
        
        # DBSCAN
        dbscan = cluster.DBSCAN(eps = eps, min_samples = min_samples, metric = metric)
        dbscan.fit(data)
        
        # Set cluster vector
        self.set_cluster_vector(dbscan.labels_)
        self.set_result(dbscan)

if __name__ == '__main__':
    example_path = "src/points.txt"
    pcd = Point_Matrix(example_path)
    pcd.load_data()

    clustering = Clustering_Operations(pcd)
    clustering.kmeans(n_clusters = 3, max_iter = 300, init = 'k-means++', algorithm = 'auto')
    print(clustering.get_cluster_vector())
    print(clustering.calculate_cluster_centers())
