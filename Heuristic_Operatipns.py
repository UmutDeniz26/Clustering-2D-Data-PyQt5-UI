import numpy as np
import random
from Data import Point_Matrix

class Heuristic_Operations(Point_Matrix):
    def __init__(self, data: Point_Matrix):
        Point_Matrix.__init__(self, data = data)

    def method_handler_heuristics(self, method_name):
        if method_name == 'Hill Climbing':
            self.hill_climbing()
        elif method_name == 'Simulated Annealing':
            self.simulated_annealing()
        else:
            print("Method not found.")

    def objective_function(self, data):
        # For example, sum of distances between points
        return np.sum(np.linalg.norm(data - np.mean(data, axis=0), axis=1))

    def generate_neighbor(self, data):
        if len(data) == 0:
            return data
        # Generate a neighbor solution by making a small random change to the current solution
        neighbor = data.copy()
        if len(data) > 1:
            idx = random.randint(0, len(data) - 1)  # Randomly select a point to modify
        else:
            idx = 0
        perturbation = np.random.uniform(-1, 1, size=2)  # Small random change
        neighbor[idx] += perturbation
        return neighbor

    def hill_climbing(self, max_iterations=1000):
        # Get data

        self.cluster_hubs = self.init_cluster_hubs()

        for i in range(max_iterations):
            # Assign new clusters
            self.assign_new_clusters()

            self.relocate_cluster_hubs()
    
            print("New value: ", self.objective_function(self.get_data_as_list()))

        print("Cluster hubs: ", self.cluster_hubs)
    
    """
    def simulated_annealing(self, max_iterations=1000, initial_temperature=100.0, cooling_rate=0.99):
        # Get data
        current_solution = np.array(self.get_data_as_list())

        current_value = self.objective_function(current_solution)

        # Initial temperature
        temperature = initial_temperature

        # Keep track of the best solution
        best_solution = current_solution
        best_value = current_value

        for iteration in range(max_iterations):
            # Generate a neighbor
            candidate_solution = self.generate_neighbor(current_solution)
            candidate_value = self.objective_function(candidate_solution)

            # Calculate the difference in objective values
            diff = candidate_value - current_value

            # Calculate the acceptance probability
            if diff < 0 or np.random.rand() < np.exp(-diff / temperature):
                current_solution = candidate_solution
                current_value = candidate_value

                # Check for a new best solution
                if candidate_value < best_value:
                    best_solution = candidate_solution
                    best_value = candidate_value

            # Cool down the temperature
            temperature *= cooling_rate

        return {"solution": best_solution, "value": best_value}
    """
    
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
    def assign_new_clusters(self):
        for point in self.get_data():
            point.set_cluster_id(self.optimal_cluster(point, self.cluster_hubs))
        
    
    # Relocate cluster hubs randomly
    def relocate_cluster_hubs(self):
        cluster_items = self.get_cluster_items()

        for key, value in cluster_items.items():
            self.cluster_hubs[key] = random.choice(value)
    
        

if __name__ == '__main__':
    pcd = Point_Matrix("src/points.txt")
    pcd.load_data()

    heuristic_ops = Heuristic_Operations(pcd)
    heuristic_ops.method_handler_heuristics('Hill Climbing')
