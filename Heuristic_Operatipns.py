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
        if len(data) == 0:
            return float('inf')
        # Example objective function: minimize the sum of distances to the origin (0, 0)
        origin = np.array([0, 0])
        return np.sum(np.sqrt(np.sum((data - origin) ** 2, axis=1)))

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

    def hill_climbing(self):
        # Get data
        data = np.array(self.get_data_as_list())

        # Initial solution
        current_solution = data
        current_value = self.objective_function(current_solution)

        # Parameters
        max_iterations = 1000
        no_improvement_limit = 100

        no_improvement_counter = 0

        for iteration in range(max_iterations):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_value = self.objective_function(neighbor)

            if neighbor_value < current_value:
                current_solution = neighbor
                current_value = neighbor_value
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= no_improvement_limit:
                break
            
        return { "solution": current_solution, "value": current_value }

    def simulated_annealing(self):
        # Get data
        data = np.array(self.get_data_as_list())

        # Initial solution
        current_solution = data
        current_value = self.objective_function(current_solution)

        # Parameters
        max_iterations = 1000
        no_improvement_limit = 100
        initial_temperature = 100.0
        cooling_rate = 0.99

        no_improvement_counter = 0
        temperature = initial_temperature

        for iteration in range(max_iterations):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_value = self.objective_function(neighbor)

            if neighbor_value < current_value:
                current_solution = neighbor
                current_value = neighbor_value
                no_improvement_counter = 0
            else:
                delta = neighbor_value - current_value
                acceptance_probability = np.exp(-delta / temperature)
                if random.random() < acceptance_probability:
                    current_solution = neighbor
                    current_value = neighbor_value
                    no_improvement_counter = 0

            temperature *= cooling_rate

            if no_improvement_counter >= no_improvement_limit:
                break


        return { "solution": current_solution, "value": current_value }

if __name__ == '__main__':
    pcd = Point_Matrix("src/points.txt")
    pcd.load_data()

    heuristic_ops = Heuristic_Operations(pcd)
    heuristic_ops.method_handler_heuristics('Hill Climbing')
