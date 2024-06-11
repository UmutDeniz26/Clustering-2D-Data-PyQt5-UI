import numpy as np

# Class to represent a point in 3D space
ID = 0

class Point:
    def __init__(self, x, y,cluster_id=None):
        self.set_coordinates(x, y)
        self.set_cluster_id(cluster_id)
        self.set_incremental_id()

    def __str__(self):
        return f"({self.__x}, {self.__y}), Cluster: {self.get_cluster_id()}\n"

    def set_cluster_id(self, cluster_id):
        self.__cluster_id = cluster_id

    def get_cluster_id(self):
        return self.__cluster_id
    
    def set_coordinates(self, x, y):
        self.__x = x
        self.__y = y

    def set_incremental_id(self):
        global ID
        self.id = ID
        ID += 1

    def get_id(self):
        return self.id
    
    def get_coordinates(self):
        return self.__x, self.__y

# Class to represent a matrix of points
class Point_Matrix:
    def __init__(self, filename= None, data=None):
        self.set_filename(filename)
        self.set_data(data)
        self.set_cluster_vector([])
        
    ######################### FILE OPERATIONS #########################

    def load_data(self, filename=None):
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
        self.filename = filename

    def get_filename(self):
        return self.filename
            
    ######################### CLUSTER OPERATIONS #########################
    
    def set_cluster_vector(self, cluster_vector):
        for i, cluster_id in enumerate(cluster_vector):
            self.data[i].set_cluster_id(cluster_id)

    def get_cluster_vector(self):
        return [point.get_cluster_id() for point in self.get_data()]

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

    # Return the number of clusters
    def get_cluster_count(self):
        return max(self.get_cluster_vector()) + 1

    # Function to get center nodes
    def calculate_center_nodes(self):        
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
        unique_labels = np.unique(self.get_cluster_vector())
        data_points = np.array(self.get_data_as_list())
        centers = np.array([data_points[self.get_cluster_vector() == label].mean(axis=0) for label in unique_labels])
        return centers
    
    # Return the all possible pairs
    def calculate_all_possible_pairs(self):
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
        return self.data
    
    def get_data_as_list(self):
        data = []
        for point in self.get_data():
            data.append(point.get_coordinates())
        return data

    def clear_data(self):
        self.data = []

    def save_data(self, solution=None, filename=None):
        
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

if __name__ == '__main__':
    pcd = Point_Matrix()
    pcd.load_data("src/points.txt")
