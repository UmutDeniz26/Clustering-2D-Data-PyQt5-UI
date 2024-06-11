
import sys
from PyQt5 import QtWidgets
from UI_Interface import UI_Interface

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = UI_Interface(template_path='Interface.ui')
    sys.exit(app.exec_())


"""
class-> UI Interface( QMainWindow, Clustering_Operations, Heuristic_Operations )
    def __init__(self, template_path)
    def init_ui(self)
    def get_buttons(self)
    def change_buttons_state(self, state, visible)
    def manual_run_clicked(self)
    def plot_initial_solution(self)
    def plot_final_solution(self)
    def clear_final_solution(self)
    def clear_initial_solution(self)
    def update_history(self, pixmap, history, index)
    def print_cluster_information(self, ret=None)
    def add_data_infromation_panel(self, data)
    def add_data_results_panel(self, data)
    def clear_data_results_panel(self)
    def clear_data_information_panel(self)
    def clustering_button_clicked(self)
    def heuristics_button_clicked(self)
    def sidebar_button_clicked(self)
    def init_side_buttons(self)
    def change_side_buttons_visibility(self, sender)
    def load_data_button(self)
    def initial_solution_button_clicked(self)
    def final_solution_button_clicked(self)
    def exit_app(self)
    def plot_to_pixmap(self, fig, label_size)
    def cv2_to_pixmap(self, img)
    def init_figure(self)
"""

"""
class -> Point_Matrix()
    def __init__(self, filename= None, data=None):
    def load_data(self, filename=None):
    def set_filename(self, filename):
    def get_filename(self):
    def set_cluster_vector(self, cluster_vector):
    def get_cluster_vector(self):
    def get_cluster_items(self):
    def get_cluster_count(self):
    def calculate_center_nodes(self):
    def calculate_distances_from_center(self):
    def calculate_cluster_centers(self):
    def calculate_all_possible_pairs(self):
    def calculate_pair_objectives(self):
    def set_data(self, data):
    def get_data(self):
    def get_data_as_list(self):
    def clear_data(self):
    def save_data(self, solution=None, filename=None):
    
class -> Point()
    def __init__(self, x, y,cluster_id=None):
    def __str__(self):
    def set_cluster_id(self, cluster_id):
    def get_cluster_id(self):
    def set_coordinates(self, x, y):
    def set_incremental_id(self):
    def get_id(self):
    def get_coordinates(self):
"""


"""
class -> Heuristic_Operations( Point_Matrix )
    def __init__(self, data: Point_Matrix):
    def method_handler_heuristics(self, method_name, arg_dict):
    def hill_climbing(self, max_iterations=1000, n_clusters=3, swap_nodes_chance=0.1, reallocate_node_chance=0.1):
    def simulated_annealing(self, max_iterations=1000, initial_temperature=100.0, cooling_rate=0.99, n_clusters=3, swap_nodes_chance=0.1, reallocate_node_chance=0.1):
    def constant_cluster_hubs_calculation(self, cluster_hubs):
    def objective_function(self, data, cluster_hubs):
    def init_cluster_hubs(self, n_hubs=3):
    def optimal_cluster(self, data, cluster_hubs):
    def assign_new_clusters(self, cluster_hubs):
    def relocate_cluster_hubs(self):
    def swap_nodes(self, data, from_id = None, to_id = None ):
    def reallocate_node(self, data):
"""

"""
class-> Clustering_Operations( Point_Matrix ):
    def __init__(self, data: Point_Matrix):
    def method_handler_clustering(self,method_name, args_dict):
    def kmeans(self, n_clusters = 3, max_iter = 300, init = 'k-means++', algorithm = 'auto'):
    def affinity_propagation(self, damping = 0.5, max_iter = 200, convergence_iter = 15):
    def mean_shift(self, bandwidth = 250, max_iter = 300):
    def spectral_clustering(self, n_clusters = 8, assign_labels = 'kmeans', eigen_solver = None, random_state = None):
    def hierarchical_clustering(self, n_clusters = 2, linkage = 'ward', distance_threshold = None):
    def dbscan(self, eps = 0.5, min_samples = 5, metric = 'euclidean'):
"""