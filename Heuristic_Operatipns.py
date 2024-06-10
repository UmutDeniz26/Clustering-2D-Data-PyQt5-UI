import numpy as np
import random
from Data import Point_Matrix
from Data import Point

class Heuristic_Operations(Point_Matrix):
    def __init__(self, data: Point_Matrix):
        Point_Matrix.__init__(self, data = data)

    def method_handler_heuristics(self, method_name, arg_dict):
        if method_name == 'Hill Climbing':
            return self.hill_climbing(arg_dict['max_iterations'])
        elif method_name == 'Simulated Annealing':
            return self.simulated_annealing(arg_dict['max_iterations'], arg_dict['initial_temperature'], arg_dict['cooling_rate'])
        else:
            print("Method not found.")

    def objective_function(self, data, cluster_hubs):
        # Calculate the sum of the distances from each point to the nearest cluster hub
        total_distance = 0
        for point in data:
            its_hub = cluster_hubs[point.get_cluster_id()]
            total_distance += np.linalg.norm(np.array(point.get_coordinates()) - np.array(its_hub.get_coordinates()), axis=0)
        return total_distance


    def hill_climbing(self, max_iterations=1000):
        # Get data

        # Initialize cluster hubs and get initial solution value
        self.cluster_hubs = self.init_cluster_hubs();self.assign_new_clusters(self.cluster_hubs)
        objective_score = self.objective_function( self.get_data(), self.cluster_hubs)

        no_change_count = 0

        for i in range(max_iterations):
            # Assign new clusters
            temp_cluster_hubs = self.relocate_cluster_hubs()
            self.assign_new_clusters(self.cluster_hubs)

            self.swap_nodes(self.get_data()) if random.random() < 0.1 else None
            self.reallocate_nodes(self.get_data()) if random.random() < 0.1 else None

            if self.objective_function( self.get_data(), temp_cluster_hubs) < objective_score:
                # Update the cluster hubs and the objective score
                self.cluster_hubs = temp_cluster_hubs
                objective_score = self.objective_function( self.get_data(), self.cluster_hubs)
                
                no_change_count = 0
            else:
                no_change_count += 1
                
            if no_change_count > 100:
                break

        self.assign_new_clusters(self.cluster_hubs)

        return {"cluster_hubs": [hub.get_id() for hub in self.cluster_hubs], "value": self.objective_function( self.get_data(), self.cluster_hubs)}

    def simulated_annealing(self, max_iterations=1000, initial_temperature=100.0, cooling_rate=0.99):
        # Get data
        data = self.get_data()
        # Initialize cluster hubs and get initial solution value
        self.cluster_hubs = self.init_cluster_hubs();self.assign_new_clusters(self.cluster_hubs)
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

            self.swap_nodes(self.get_data()) if random.random() < 0.1 else None
            self.reallocate_nodes(self.get_data()) if random.random() < 0.1 else None

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

        return {"cluster_hubs": [hub.get_id() for hub in best_cluster_hubs], "value": best_objective_score}

    # Manually hub selection
    def constant_cluster_hubs_calculation(self, cluster_hubs):
        
        if type(cluster_hubs[0]) != Point:
            cluster_hubs = [ hub for hub in self.get_data() if hub.get_id() in cluster_hubs]

        # Get data
        data = self.get_data()
        # Assign new clusters
        self.assign_new_clusters(cluster_hubs)
        # Calculate the objective score
        return self.objective_function(data, cluster_hubs)


    # Initialize cluster hubs
    def init_cluster_hubs(self, n_hubs=3):
        return np.random.choice(self.get_data(), n_hubs, replace=False)
    

    # Calculate the distance between two points
    def optimal_cluster(self, data, cluster_hubs):
        distances = []
        for i in range(len(cluster_hubs)):
            distances.append(np.linalg.norm(np.array(data.get_coordinates()) - np.array(cluster_hubs[i].get_coordinates()), axis=0))
        return np.argmin(distances, axis=0)
    
    
    # Assign new clusters
    def assign_new_clusters(self, cluster_hubs):
        for point in self.get_data():
            point.set_cluster_id(self.optimal_cluster(point, cluster_hubs))
        
    
    # Relocate cluster hubs randomly
    def relocate_cluster_hubs(self):
        # Get data
        cluster_items = self.get_cluster_items()

        temp_hubs = [0]*3
        for key, value in cluster_items.items():
            temp_hubs[key] = random.choice(value)
    
        return temp_hubs    
    
    def swap_nodes(self, data):
        # Random i and j values, non-hub
        possible_indexes = [i for i in range(len(data)) if data[i].get_id() not in [hub.get_id() for hub in self.cluster_hubs]]
        
        # Random i and j values
        i = random.choice(possible_indexes)
        j = random.choice(possible_indexes)

        # Swap the nodes
        temp = data[i].get_cluster_id()
        data[i].set_cluster_id( data[j].get_cluster_id() )
        data[j].set_cluster_id( temp )
                
    def reallocate_nodes(self, data):
        # Random i value, non-hub
        possible_indexes = [i for i in range(len(data)) if data[i].get_id() not in [hub.get_id() for hub in self.cluster_hubs]]
        
        # Random i value
        i = random.choice(possible_indexes)

        # Random j value, hub ===> from 0 to len(cluster_hubs)
        j = random.choice(range(len(self.cluster_hubs)))

        # Reallocation
        data[i].set_cluster_id(j)


if __name__ == '__main__':
    pcd = Point_Matrix("src/points.txt")
    pcd.load_data()

    heuristic_ops = Heuristic_Operations(pcd)
    heuristic_ops.method_handler_heuristics('Hill Climbing')
