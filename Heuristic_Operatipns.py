import numpy as np
import random
from Data import Point_Matrix
from Data import Point

class Heuristic_Operations(Point_Matrix):
    """
    @brief Class that contains the heuristic methods for clustering.
    @details The class contains the following heuristic methods:
    """
    def __init__(self, data: Point_Matrix):
        """
        @brief Constructor for the Heuristic_Operations class.
        @param data: The data to be clustered.
        """
        Point_Matrix.__init__(self, data = data)

    def method_handler_heuristics(self, method_name, arg_dict):
        """
        @brief Method that handles the heuristic methods.
        @param method_name: The name of the method to be used.
        @param arg_dict: The arguments for the method.
        @return The result of the method.
        """

        if method_name == 'Hill Climbing':
            return self.hill_climbing(
                arg_dict['max_iterations'], arg_dict['n_clusters'], arg_dict['swap_nodes_chance'], arg_dict['reallocate_node_chance']
                )
        elif method_name == 'Simulated Annealing':
            return self.simulated_annealing(
                arg_dict['max_iterations'], arg_dict['initial_temperature'], arg_dict['cooling_rate'], arg_dict['n_clusters'],
                arg_dict['swap_nodes_chance'], arg_dict['reallocate_node_chance']
                )
        else:
            print("Method not found.")

    ############################# HEURISTIC METHODS #############################

    def hill_climbing(self, max_iterations=1000, n_clusters=3, swap_nodes_chance=0.1, reallocate_node_chance=0.1):
        """
        @brief Method that performs the Hill Climbing heuristic.
        @param max_iterations: The maximum number of iterations.
        @param n_clusters: The number of clusters.
        @param swap_nodes_chance: The chance of swapping nodes.
        @param reallocate_node_chance: The chance of reallocating a node.
        @return The result of the Hill Climbing heuristic.
        """
        
        # Initialize cluster hubs and get initial solution value
        self.cluster_hubs = self.init_cluster_hubs( n_clusters )
        self.assign_new_clusters(self.cluster_hubs)
        objective_score = self.objective_function( self.get_data(), self.cluster_hubs )

        no_change_count = 0

        for i in range(max_iterations):
            # Assign new clusters
            temp_cluster_hubs = self.relocate_cluster_hubs()
            self.assign_new_clusters(temp_cluster_hubs)

            self.swap_nodes(self.get_data()) if random.random() < swap_nodes_chance else None
            self.reallocate_node(self.get_data()) if random.random() < reallocate_node_chance else None

            # Calculate the new objective score
            new_objective_score = self.objective_function(self.get_data(), temp_cluster_hubs)

            # If the new solution is better, update the current solution
            if new_objective_score < objective_score:
                objective_score = new_objective_score
                self.cluster_hubs = temp_cluster_hubs
                no_change_count = 0
            else:
                no_change_count += 1

            if no_change_count > 100:
                break

        self.assign_new_clusters(self.cluster_hubs)

        return {"Cluster Hubs": [hub.get_id() for hub in self.cluster_hubs], "Best Objective Score": objective_score}

    def simulated_annealing(self, max_iterations=1000, initial_temperature=100.0, cooling_rate=0.99, n_clusters=3, swap_nodes_chance=0.1, reallocate_node_chance=0.1):
        """
        @brief Method that performs the Simulated Annealing heuristic.
        @param max_iterations: The maximum number of iterations.
        @param initial_temperature: The initial temperature.
        @param cooling_rate: The cooling rate.
        @param n_clusters: The number of clusters.
        @param swap_nodes_chance: The chance of swapping nodes.
        @param reallocate_node_chance: The chance of reallocating a node.
        @return The result of the Simulated Annealing heuristic.
        """
        
        # Get data
        data = self.get_data()
        # Initialize cluster hubs and get initial solution value
        self.cluster_hubs = self.init_cluster_hubs( n_clusters )
        self.assign_new_clusters(self.cluster_hubs)
        objective_score = self.objective_function(data, self.cluster_hubs)

        # Initialize the best solution
        best_cluster_hubs = self.cluster_hubs
        best_objective_score = objective_score

        # Initialize the current solution
        current_cluster_hubs = self.cluster_hubs
        current_objective_score = objective_score

        # Initialize the temperature
        temperature = initial_temperature

        for i in range(max_iterations):
            # Assign new clusters
            temp_cluster_hubs = self.relocate_cluster_hubs()
            self.assign_new_clusters(self.cluster_hubs)

            self.swap_nodes(self.get_data()) if random.random() < swap_nodes_chance else None
            self.reallocate_node(self.get_data()) if random.random() < reallocate_node_chance else None

            # Calculate the new objective score
            new_objective_score = self.objective_function(data, temp_cluster_hubs)

            # Calculate the acceptance probability
            acceptance_probability = np.exp((current_objective_score - new_objective_score) / temperature)

            # If the new solution is better, update the current solution and the best solution
            if new_objective_score < current_objective_score or random.random() < acceptance_probability:
                current_objective_score = new_objective_score

                if new_objective_score < best_objective_score:
                    best_cluster_hubs = temp_cluster_hubs
                    best_objective_score = new_objective_score

            # Update the temperature
            temperature *= cooling_rate

        self.assign_new_clusters(best_cluster_hubs)
        self.cluster_hubs = best_cluster_hubs

        return {"Cluster Hubs": [hub.get_id() for hub in best_cluster_hubs], "Best Objective Score": best_objective_score}

    # Manually hub selection
    def constant_cluster_hubs_calculation(self, cluster_hubs):
        """
        @brief Method that calculates the objective score for a given set of cluster hubs.
        @param cluster_hubs: The cluster hubs.
        @return The objective score for the given set of cluster hubs.
        """

        if type(cluster_hubs[0]) != Point:
            cluster_hubs = [ hub for hub in self.get_data() if hub.get_id() in cluster_hubs]

        # Get data
        data = self.get_data()
        # Assign new clusters
        self.assign_new_clusters(cluster_hubs)
        # Calculate the objective score
        return self.objective_function(data, cluster_hubs)

    ############################# HEURISTIC AUXILIARY METHODS #############################

    def objective_function(self, data, cluster_hubs):
        """
        @brief Method that calculates the objective score for a given set of cluster hubs.
        @param data: The data.
        @param cluster_hubs: The cluster hubs.
        @return The objective score for the given set of cluster hubs.
        """
        
        # Calculate the sum of the distances from each point to the nearest cluster hub
        total_distance = 0
        for point in data:
            its_hub = cluster_hubs[point.get_cluster_id()]
            total_distance += np.linalg.norm(np.array(point.get_coordinates()) - np.array(its_hub.get_coordinates()), axis=0)
        return total_distance

    # Initialize cluster hubs
    def init_cluster_hubs(self, n_hubs=3):
        """
        @brief Method that initializes the cluster hubs.
        @param n_hubs: The number of cluster hubs.
        @return The initialized cluster hubs.
        """

        return np.random.choice(self.get_data(), n_hubs, replace=False)
    

    # Calculate the best cluster for a point
    def optimal_cluster(self, data, cluster_hubs):
        """
        @brief Method that calculates the best cluster for a given point.
        @param data: The point.
        @param cluster_hubs: The cluster hubs.
        @return The best cluster for the given point.
        """

        distances = []
        for i in range(len(cluster_hubs)):
            distances.append(np.linalg.norm(np.array(data.get_coordinates()) - np.array(cluster_hubs[i].get_coordinates()), axis=0))
        return np.argmin(distances, axis=0)
    
    
    # Assign new clusters
    def assign_new_clusters(self, cluster_hubs):
        """
        @brief Method that assigns new clusters to the data.
        @param cluster_hubs: The cluster hubs.
        """
        for point in self.get_data():
            point.set_cluster_id(self.optimal_cluster(point, cluster_hubs))
        

    ############################# HEURISTIC Random Operations #############################

    # Relocate cluster hubs randomly
    def relocate_cluster_hubs(self):
        """
        @brief Method that relocates the cluster hubs randomly.
        @return The relocated cluster hubs.
        """
        
        # Get data
        cluster_items = self.get_cluster_items()

        temp_hubs = [0]*len(self.cluster_hubs)
        for key, value in cluster_items.items():
            temp_hubs[key] = random.choice(value)
    
        return temp_hubs    
    
    def swap_nodes(self, data, from_id = None, to_id = None ):
        """
        @brief Method that swaps two nodes.
        @param data: The data.
        """

        # Random i and j values, non-hub
        possible_indexes = [i for i in range(len(data)) if data[i].get_id() not in [hub.get_id() for hub in self.cluster_hubs]]
        
        # Random i and j values
        i = random.choice(possible_indexes)
        j = random.choice(possible_indexes)

        if from_id != None and to_id != None:
            i = from_id
            j = to_id

        # Swap the nodes
        temp = data[i].get_cluster_id()
        data[i].set_cluster_id( data[j].get_cluster_id() )
        data[j].set_cluster_id( temp )
                
    def reallocate_node(self, data):
        """
        @brief Method that reallocates a node.
        @param data: The data.
        """

        # Random i value, non-hub
        possible_indexes = [i for i in range(len(data)) if data[i].get_id() not in [hub.get_id() for hub in self.cluster_hubs]]
        
        # Random i value
        i = random.choice(possible_indexes)

        # Random j value, hub ===> from 0 to len(cluster_hubs)
        j = random.choice(range(len(self.cluster_hubs)))

        # Reallocation
        data[i].set_cluster_id(j)