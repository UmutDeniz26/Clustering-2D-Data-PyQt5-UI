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
            return self.dbscan(
                eps = args_dict['eps'], min_samples = args_dict['min_samples'], metric = args_dict['metric']
                )
        else:
            print("Method not found.")


    ####################################### GETTERS ####################################### 

    def get_cluster_count(self):
        return max(self.get_cluster_vector()) + 1

    # Function to get center nodes
    def get_center_nodes(self):        
        # Minimum distance between a point and a cluster, will be the center of the cluster
        cluster_centers = self.calculate_cluster_centers()
        self.center_nodes = []
        
        # Get center nodes
        for i in range(self.get_cluster_count()):
            # Get points in cluster i
            temp = []
            for point in self.get_data():
                if point.get_cluster_id() == i:
                    distance = np.linalg.norm(point.get_coordinates() - cluster_centers[i])
                    temp.append((point, distance))
            
            temp.sort(key = lambda x: x[1])
            
            center_node = temp[0][0]
            self.center_nodes.append(center_node)

        return self.center_nodes

    # Function to get cluster items
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
    
    def calculate_distances_from_center(self):
        # Get data
        centers = self.calculate_cluster_centers()
        center_nodes = self.get_center_nodes()

        # Initialize distances_from_center
        distances_from_center = {} # {point_id: distance, point_id: distance, ...}

        # Calculate distances from center
        for i, center_node in enumerate(center_nodes):
            center = centers[i]
            distance = np.linalg.norm(center_node.get_coordinates() - center)
            distances_from_center.update({center_node.get_id(): distance})

        return distances_from_center

    def calculate_all_possible_pairs(self):
        cluster_nodes = self.get_center_nodes()

        # Initialize all_possible_pairs
        all_possible_pairs = []
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                if i != j and (cluster_nodes[i], cluster_nodes[j]) not in all_possible_pairs:
                    all_possible_pairs.append((cluster_nodes[i], cluster_nodes[j]))


        return all_possible_pairs

    def calculate_pair_objectives(self):
        # Then, for each pair calculate the following objective 
        # function:
        # OBJij = Dihi + 0.75 * Dhihj + Djhj
        # where dihi distance of farthest point in the cluster i, dhihj diatnce between hub i and hub j and djhj
        # distance of farthest point in the cluster j. Also, consider 2*max(dihi). Lastly, get the maximum of pair 
        # objectives as an objective function result.
        
        # Get data
        all_pairs = self.calculate_all_possible_pairs()

        # Initialize pair_objectives
        pair_objectives = {} # {(point_id, point_id): objective, (point_id, point_id): objective, ...}

        # Calculate pair objectives
        for pair in all_pairs:
            # Get cluster nodes
            cluster_i, cluster_j = pair
            cluster_i_points = [point for point in self.get_data() if point.get_cluster_id() == cluster_i.get_cluster_id()]
            cluster_j_points = [point for point in self.get_data() if point.get_cluster_id() == cluster_j.get_cluster_id()]

            # Get distances
            distances_i = [np.linalg.norm(np.array(point.get_coordinates()) - np.array(cluster_i.get_coordinates())) for point in cluster_i_points]
            distances_j = [np.linalg.norm(np.array(point.get_coordinates()) - np.array(cluster_j.get_coordinates())) for point in cluster_j_points]

            # Get maximum distances
            dihi = max(distances_i)
            djhj = max(distances_j)

            # Get distance between hubs
            dhihj = np.linalg.norm(np.array(cluster_i.get_coordinates()) - np.array(cluster_j.get_coordinates()))

            # Calculate objective
            objective = dihi + 0.75 * dhihj + djhj

            # Update pair_objectives
            pair_objectives.update({(cluster_i.get_id(), cluster_j.get_id()): objective})
    
        return pair_objectives, max(pair_objectives.values())

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
        
    def mean_shift(self, bandwidth = 250, max_iter = 300):
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
                        self.set_result(dbscan)
                        return {"eps": eps, "min_samples": min_samples, "metric": metric, "auto": True}
        else:
            self.set_cluster_vector(dbscan.labels_)
            self.set_result(dbscan)
            return {"eps": eps, "min_samples": min_samples, "metric": metric, "auto": False}

if __name__ == '__main__':
    example_path = "src/points.txt"
    pcd = Point_Matrix(example_path)
    pcd.load_data()

    clustering = Clustering_Operations(pcd)
    clustering.dbscan(eps = 200, min_samples = 5, metric = 'euclidean')
    print(clustering.get_cluster_vector())
    
    #clustering.kmeans(n_clusters = 3, max_iter = 300, init = 'k-means++', algorithm = 'auto')
    #exit()
    print(clustering.calculate_distances_from_center())
    # all possible pairs
    print([ (x.get_id(), y.get_id()) for x, y in clustering.calculate_all_possible_pairs()])
    #print(clustering.calculate_pair_objectives())