import numpy as np

# Class to represent a point in 3D space
ID = 0

class Point:
    """
    @brief Class to represent a point in 2D space
    @details This class is used to represent a point in 2D space. It has x and y coordinates and a cluster id.
    """
    def __init__(self, x, y,cluster_id=None):
        """
        @brief Constructor for Point class
        @param x X coordinate of the point
        @param y Y coordinate of the point
        @param cluster_id Cluster id of the point
        @details This function initializes the Point object with the given x and y coordinates and cluster id.
        """
        self.set_coordinates(x, y)
        self.set_cluster_id(cluster_id)
        self.set_incremental_id()

    def __str__(self):
        """
        @brief String representation of the Point object
        @details This function returns the string representation of the Point object.
        @return String representation of the Point object
        """
        return f"({self.__x}, {self.__y}), Cluster: {self.get_cluster_id()}\n"

    def set_cluster_id(self, cluster_id):
        """
        @brief Function to set the cluster id of the point
        @param cluster_id Cluster id of the point
        """
        self.__cluster_id = cluster_id

    def get_cluster_id(self):
        """
        @brief Function to get the cluster id of the point
        @return Cluster id of the point
        """
        return self.__cluster_id
    
    def set_coordinates(self, x, y):
        """
        @brief Function to set the coordinates of the point
        @param x X coordinate of the point
        @param y Y coordinate of the point
        """
        self.__x = x
        self.__y = y

    def set_incremental_id(self):
        """
        @brief Function to set the incremental ID of the point
        """
        global ID
        self.id = ID
        ID += 1

    def get_id(self):
        """
        @brief Function to get the ID of the point
        @return ID of the point
        """
        return self.id
    
    def get_coordinates(self):
        """
        @brief Function to get the coordinates of the point
        @return X and Y coordinates of the point
        """
        return self.__x, self.__y

# Class to represent a matrix of points
class Point_Matrix:
    """
    @brief Class to represent a matrix of points in 2D space
    @details This class is used to represent a matrix of points in 2D space. It has a list of points and a filename. 
    """
    def __init__(self, filename= None, data=None):
        """
        @brief Constructor for Point_Matrix class
        @param filename Filename of the data file
        @param data Data to be loaded
        @details This function initializes the Point_Matrix object with the given filename and data.
        """

        self.set_filename(filename)
        self.set_data(data)
        self.set_cluster_vector([])
        
    ######################### FILE OPERATIONS #########################

    def load_data(self, filename=None):
        """
        @brief Function to load data from a file
        @param filename Filename of the data file
        @details This function reads data from the file with the given filename. It reads the x and y coordinates of the points and cluster id if provided.
        """
        # Set filename if provided
        if filename is not None:
            self.set_filename(filename)

        # Clear data
        self.clear_data()
        
        try:
            # Read data from file
            with open(self.get_filename(), 'r') as file:
                data = file.read()

                data = data.split("\n")
                for line in data:
                    if line:
                        line_elements = line.split(" ")
                        x,y = line_elements[0], line_elements[1]
                        
                        cluster_id = line_elements[2] if len(line_elements) > 2 else None 
                        
                        self.data.append(Point(float(x), float(y), cluster_id))

                print("Data loaded successfully: ")
        except FileNotFoundError:
            print("File not found.")

    def set_filename(self, filename):
        """
        @brief Function to set the filename of the data file
        @param filename Filename of the data file
        """

        self.filename = filename

    def get_filename(self):
        """
        @brief Function to get the filename of the data file
        @return Filename of the data file
        """

        return self.filename
            
    ######################### CLUSTER OPERATIONS #########################
    
    def set_cluster_vector(self, cluster_vector):
        """
        @brief Function to set the cluster vector of the points
        @param cluster_vector Cluster vector of the points
        """
        for i, cluster_id in enumerate(cluster_vector):
            self.data[i].set_cluster_id(cluster_id)

    def get_cluster_vector(self):
        """
        @brief Function to get the cluster vector of the points
        @return Cluster vector of the points
        """
        return [point.get_cluster_id() for point in self.get_data()]

    # Function to get cluster items
    def get_cluster_items(self):
        """
        @brief Function to get the cluster items
        @return Cluster items
        """
        
        # Get cluster items
        count_clusters = max(self.get_cluster_vector()) + 1

        cluster_items = {i: [] for i in range(count_clusters)}
        points = self.get_data()

        for i in range(count_clusters):
            for point in points:
                if point.get_cluster_id() == i:
                    cluster_items[i].append(point)
                    
        return cluster_items

    # Return the number of clusters
    def get_cluster_count(self):
        """
        @brief Function to get the number of clusters
        @return Number of clusters
        """
        
        return max(self.get_cluster_vector()) + 1

    # Function to get center nodes
    def calculate_center_nodes(self):        
        """
        @brief Function to calculate the center nodes
        @return Center nodes
        """
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

    # Return the center nodes
    def calculate_distances_from_center(self):
        """
        @brief Function to calculate the distances from the center
        @return Distances from the center
        """
        
        # Get data
        centers = self.calculate_cluster_centers()
        center_nodes = self.calculate_center_nodes()

        # Initialize distances_from_center
        distances_from_center = {} # {point_id: distance, point_id: distance, ...}

        # Calculate distances from center
        for i, center_node in enumerate(center_nodes):
            center = centers[i]
            distance = np.linalg.norm(center_node.get_coordinates() - center)
            distances_from_center.update({center_node.get_id(): distance})

        return distances_from_center


    # Function to calculate cluster centers for other methods
    def calculate_cluster_centers(self):
        """
        @brief Function to calculate the cluster centers
        @return Cluster centers
        """

        unique_labels = np.unique(self.get_cluster_vector())
        data_points = np.array(self.get_data_as_list())
        centers = np.array([data_points[self.get_cluster_vector() == label].mean(axis=0) for label in unique_labels])
        return centers
    
    # Return the all possible pairs
    def calculate_all_possible_pairs(self):
        """
        @brief Function to calculate all possible pairs
        @return All possible pairs
        """
        
        cluster_nodes = self.calculate_center_nodes()

        # Initialize all_possible_pairs
        all_possible_pairs = []
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                if i != j and (cluster_nodes[i], cluster_nodes[j]) not in all_possible_pairs:
                    all_possible_pairs.append((cluster_nodes[i], cluster_nodes[j]))


        return all_possible_pairs

    # Function to calculate the objective function for each pair
    def calculate_pair_objectives(self):
        """
        @brief Function to calculate the objective function for each pair
        @return Pair objectives
        """

        # Get pairs
        all_pairs = self.calculate_all_possible_pairs()

        # Initialize pair_objectives
        pair_objectives = {} # {(point_id, point_id): objective, (point_id, point_id): objective, ...}

        # Calculate pair objectives
        for pair in all_pairs:
            # Get non-hub cluster nodes
            cluster_i, cluster_j = pair

            # Get distances from centers
            distances_from_centers = self.calculate_distances_from_center()
            dihi, djhj = distances_from_centers[cluster_i.get_id()], distances_from_centers[cluster_j.get_id()]

            # Get distance between hubs
            dhihj = np.linalg.norm(np.array(cluster_i.get_coordinates()) - np.array(cluster_j.get_coordinates()))

            # Calculate objective
            objective = dihi + 0.75 * dhihj + djhj

            # Update pair_objectives
            pair_objectives.update({(cluster_i.get_id(), cluster_j.get_id()): objective})
    
        return pair_objectives, max(pair_objectives.values())



    ######################### DATA OPERATIONS #########################

    def set_data(self, data):
        """
        @brief Function to set the data of the Point_Matrix object
        @param data Data to be set
        """
        
        if isinstance(data, list):
            for point in data:
                print(point)
                self.data.append(Point(point[0], point[1]))
        
        elif isinstance(data, Point_Matrix):
            self.data = data.get_data()
        
        elif isinstance(data, str):
            self.load_data(data)

        elif data is None:
            self.clear_data()

    def get_data(self):
        """
        @brief Function to get the data of the Point_Matrix object
        @return Data of the Point_Matrix object
        """
        return self.data
    
    def get_data_as_list(self):
        """
        @brief Function to get the data of the Point_Matrix object as a list
        @return Data of the Point_Matrix object as a list
        """
        data = []
        for point in self.get_data():
            data.append(point.get_coordinates())
        return data

    def clear_data(self):
        """
        @brief Function to clear the data of the Point_Matrix object
        """
        self.data = []

    def save_data(self, solution=None, filename=None):
        """
        @brief Function to save the data to a file
        @param solution Solution to be saved
        @param filename Filename of the file
        @details This function saves the solution to a file with the given filename.
        """
    
        # Set filename if provided
        if filename is not None:
            self.set_filename(filename) 

        if solution is None:
            solution = "\n".join([f"{x} {y}" for x, y in self.get_data_as_list()])
        else:
            solution = "\n".join([f"{x} {y}" for x, y in solution])

        try:
            with open(self.get_filename(), 'w') as file:
                file.write(solution)
            print("Data saved successfully.")
        except:
            print("Error occurred while saving data.")

