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
        current_solution = np.array(self.get_data_as_list())
        current_value = self.objective_function(current_solution)

        for iteration in range(max_iterations):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_value = self.objective_function(neighbor)

            if neighbor_value < current_value:
                current_solution = neighbor
                current_value = neighbor_value

            
        return { "solution": current_solution, "value": current_value }

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
    

    def relocate_hub(self):
        clusters = self.get_clusters()
        hubs = self.get_hubs()

        # Choose a random cluster that has at least one non-hub node
        eligible_clusters = [cluster for cluster in clusters if len(cluster) > 1]
        if not eligible_clusters:
            return

        cluster = random.choice(eligible_clusters)

        # Randomly choose a non-hub node from the cluster to become the new hub
        non_hub_nodes = [node for node in cluster if node not in hubs]
        new_hub = random.choice(non_hub_nodes)

        # Update the hub assignment
        old_hub = hubs[cluster]
        hubs[cluster] = new_hub

        # Move the old hub to be a regular node in the cluster
        cluster.remove(new_hub)
        cluster.append(old_hub)
        self.set_hubs(hubs)

    def reallocate_nodes(self):
        clusters = self.get_clusters()
        hubs = self.get_hubs()

        # Choose a random cluster
        cluster = random.choice(clusters)

        # Ensure the cluster is not a single hub-node
        if len(cluster) <= 1:
            return

        # Choose a random non-hub node to reallocate
        non_hub_nodes = [node for node in cluster if node not in hubs]
        node_to_reallocate = random.choice(non_hub_nodes)

        # Choose a new cluster for the node
        new_cluster = random.choice([c for c in clusters if c != cluster])

        # Move the node to the new cluster
        cluster.remove(node_to_reallocate)
        new_cluster.append(node_to_reallocate)

    def swap_nodes(self):
        clusters = self.get_clusters()
        hubs = self.get_hubs()

        # Choose two different clusters
        if len(clusters) < 2:
            return

        cluster1, cluster2 = random.sample(clusters, 2)

        # Ensure both clusters have non-hub nodes
        non_hub_nodes1 = [node for node in cluster1 if node not in hubs]
        non_hub_nodes2 = [node for node in cluster2 if node not in hubs]

        if not non_hub_nodes1 or not non_hub_nodes2:
            return

        # Choose a random non-hub node from each cluster
        node1 = random.choice(non_hub_nodes1)
        node2 = random.choice(non_hub_nodes2)

        # Swap the nodes
        cluster1.remove(node1)
        cluster2.remove(node2)

        cluster1.append(node2)
        cluster2.append(node1)

if __name__ == '__main__':
    pcd = Point_Matrix("src/points.txt")
    pcd.load_data()

    heuristic_ops = Heuristic_Operations(pcd)
    heuristic_ops.method_handler_heuristics('Hill Climbing')
