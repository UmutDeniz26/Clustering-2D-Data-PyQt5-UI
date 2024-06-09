# Read UI_Interface.ui file and show the window
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt5.QtWidgets import QMainWindow


from Clustering_Operations import Clustering_Operations
from Heuristic_Operatipns import Heuristic_Operations

from Get_Data_Dialog import Get_Data_Dialog
# Clustering_Operations requires Point_Matrix class from Data.py from init
class UI_Interface(QMainWindow, Clustering_Operations, Heuristic_Operations):
    def __init__(self, template_path):
        super(UI_Interface, self).__init__( data = None )

        uic.loadUi(template_path, self)
        self.show()
        
        # History elements initialization
        self.initial_solution_png_hist = []; self.initial_solution_hist_index = 0
        self.final_solution_png_hist = []; self.final_solution_hist_index = 0

        # Hide the full menu widget
        self.full_menu_widget.setVisible(False)

        # Connect buttons to functions
        self.open_data.clicked.connect(self.load_data_button)
        self.manual_run.clicked.connect(self.manual_run_clicked)

        # Initialize side menu buttons
        self.init_side_buttons()

        # Side menu buttons
        self.side_menu_buttons = [ self.initial_solution_side, self.final_solution_side, self.clustering_side, self.heuristics_side ]
        for button in self.side_menu_buttons:
            button.clicked.connect(self.sidebar_button_clicked)

        # Always display buttons
        self.always_display_buttons = [self.open_data, self.exit_button, self.menu_open_data, self.menu_exit, self.menu_file, self.manual_run]

        # Initialize the data
        self.change_buttons_state()


    ###################### UI Operations ######################

    def get_buttons(self):
        buttons = []

        for button in self.findChildren(QtWidgets.QPushButton):
            buttons.append(button)
        for button in self.findChildren(QtWidgets.QMenu):
            buttons.append(button)
        buttons.append(self.menu_save_initial_solution)
        buttons.append(self.menu_save_final_solution)

        return buttons
    
    def change_buttons_state(self):
        for button in self.get_buttons():
            if button in self.always_display_buttons:
                continue
            button.setDisabled(True) if button.isEnabled() else button.setDisabled(False)


    ###################### Manual Operations ######################


    def manual_run_clicked(self):
        # QTextEdit 
        hubs = self.manual_hubs.toPlainText()
        nodes = self.manual_nodes.toPlainText()

        print(hubs, nodes)
         
    
    #############################################################

    ###################### File Operations ######################

    def load_data_button(self):
        data_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "data.txt", "Data files (*.txt)")[0]
        if data_path:
            self.load_data(data_path)
            self.plot_initial_solution()
            self.change_buttons_state()
    
    

    ###################### Side Bar ######################
    
    def sidebar_button_clicked(self):
        
        # Get the sender
        sender = self.sender()

        # Hide the full menu widget if the sender is different
        if not hasattr(self, 'hold_sender') or self.hold_sender != sender or not self.full_menu_widget.isVisible():
            self.full_menu_widget.setVisible(True)
        else:
            self.full_menu_widget.setVisible(False)

        # Edit the full menu buttons and hold the sender
        self.change_side_buttons_visibility(sender)    
        self.hold_sender = sender

    def init_side_buttons(self):
        buttons = [
            # Initial Solution
            { 'name': 'Save As', 'object_name': 'side_initial_save_as', 'function': self.initial_solution_button_clicked },
            { 'name': 'Save', 'object_name': 'side_initial_save', 'function': self.initial_solution_button_clicked },
            { 'name': 'Export As', 'object_name': 'side_initial_export_as', 'function': self.initial_solution_button_clicked },
            { 'name': 'Undo', 'object_name': 'side_initial_undo', 'function': self.initial_solution_button_clicked },
            { 'name': 'Redo', 'object_name': 'side_initial_redo', 'function': self.initial_solution_button_clicked },

            # Final Solution
            { 'name': 'Save As', 'object_name': 'side_final_save_as', 'function': self.final_solution_button_clicked },
            { 'name': 'Save', 'object_name': 'side_final_save', 'function': self.final_solution_button_clicked },
            { 'name': 'Export As', 'object_name': 'side_final_export_as', 'function': self.final_solution_button_clicked },
            { 'name': 'Undo', 'object_name': 'side_final_undo', 'function': self.final_solution_button_clicked },
            { 'name': 'Redo', 'object_name': 'side_final_redo', 'function': self.final_solution_button_clicked },
            
            # Clustering
            { 'name': 'K-Means', 'object_name': 'side_clustering_kmeans', 'function': self.clustering_button_clicked },
            { 'name': 'Affinity Propagation', 'object_name': 'side_clustering_affinity_propagation', 'function': self.clustering_button_clicked },
            { 'name': 'Mean Shift', 'object_name': 'side_clustering_mean_shift', 'function': self.clustering_button_clicked },
            { 'name': 'Spectral Clustering', 'object_name': 'side_clustering_spectral_clustering', 'function': self.clustering_button_clicked },
            { 'name': 'Hierarchical Clustering', 'object_name': 'side_clustering_hierarchical_clustering', 'function': self.clustering_button_clicked },
            { 'name': 'DBSCAN', 'object_name': 'side_clustering_dbscan', 'function': self.clustering_button_clicked },
            
            # Heuristics
            { 'name': 'Hill Climbing', 'object_name': 'side_heuristics_hill_climbing', 'function': self.heuristics_button_clicked },
            { 'name': 'Simulated Annealing', 'object_name': 'side_heuristics_simulated_annealing', 'function': self.heuristics_button_clicked }
        ]

        # Generate buttons
        for button_dict in buttons:
            button = QtWidgets.QPushButton(button_dict['name'])
            button.setObjectName(button_dict['object_name'])
            button.clicked.connect(button_dict['function'])
            self.toolbox_layout.addWidget(button)

    def change_side_buttons_visibility(self, sender):
        
        # Hide all buttons of self.toolbox_layout
        for i in range(self.toolbox_layout.count()):
            self.toolbox_layout.itemAt(i).widget().setVisible(False)
            
        # Respect to the sender, show the buttons
        if sender == self.initial_solution_side:
            button_object_names = [ 'side_initial_save_as', 'side_initial_save', 'side_initial_export_as', 'side_initial_undo', 'side_initial_redo' ]
        elif sender == self.final_solution_side:
            button_object_names = [ 'side_final_save_as', 'side_final_save', 'side_final_export_as', 'side_final_undo', 'side_final_redo' ]
        elif sender == self.clustering_side:
            button_object_names = [ 'side_clustering_kmeans', 'side_clustering_affinity_propagation', 'side_clustering_mean_shift', 'side_clustering_spectral_clustering', 'side_clustering_hierarchical_clustering', 'side_clustering_dbscan' ]
        elif sender == self.heuristics_side:
            button_object_names = [ 'side_heuristics_hill_climbing', 'side_heuristics_simulated_annealing' ]

        for button_object_name in button_object_names:
            button = self.findChild(QtWidgets.QPushButton, button_object_name)
            button.setVisible(True)
        
    ##############################################################

    ###################### Monitor Operations ######################


    # Display functions
    def plot_initial_solution(self):
        # get plot
        initial_solution_data = self.get_data()
        label_size = self.monitor_initial_solution.width(), self.monitor_initial_solution.height()
        
        # 2D plot
        fig, ax = self.init_figure()

        for point in initial_solution_data:
            ax.scatter(point.get_coordinates()[0], point.get_coordinates()[1], c='black')
            ax.text(point.get_coordinates()[0], point.get_coordinates()[1], str(point.get_id()), fontsize=12)
        
        # Convert plot to pixmap
        pixmap = self.plot_to_pixmap(fig, label_size)
        self.monitor_initial_solution.setPixmap(pixmap)
        plt.close(fig)

        if self.initial_solution_hist_index == 0:
            self.initial_solution_png_hist.insert(0, pixmap)
        else:
            self.initial_solution_png_hist = self.initial_solution_png_hist[self.initial_solution_hist_index:]
            self.initial_solution_png_hist.insert(0, pixmap)
            self.initial_solution_hist_index = 0

    def init_figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        return fig, ax


    def plot_final_solution(self):
        
        # Get label size
        label_size = self.monitor_final_solution.width(), self.monitor_final_solution.height()
        
        # Get output data
        cluster_centers = self.calculate_cluster_centers()
        center_nodes = [ point.get_coordinates() for point in self.get_center_nodes() ]

        # Generate 2D Plot 
        fig, ax = self.init_figure()
        color_list = [ 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta' ]
        
        for point in self.get_data():
            if point.get_coordinates() in center_nodes:
                ax.scatter(point.get_coordinates()[0], point.get_coordinates()[1], c='blue')
                ax.text(point.get_coordinates()[0], point.get_coordinates()[1], str(point.get_id()), fontsize=12)
            else:
                ax.scatter(point.get_coordinates()[0], point.get_coordinates()[1], c=color_list[point.get_cluster_id()])
                ax.text(point.get_coordinates()[0], point.get_coordinates()[1], str(point.get_id()), fontsize=12)
        
        for cluster_pos in cluster_centers:
            ax.scatter(cluster_pos[0], cluster_pos[1], c='blue', s=100, marker='x')
        

        # Convert plot to pixmap
        pixmap = self.plot_to_pixmap(fig, label_size)
        self.monitor_final_solution.setPixmap(pixmap)
        plt.close(fig)

        # Append to final solution image history
        if self.final_solution_hist_index == 0:
            self.final_solution_png_hist.insert(0, pixmap)
        else:
            self.final_solution_png_hist = self.final_solution_png_hist[self.final_solution_hist_index:]
            self.final_solution_png_hist.insert(0, pixmap)
            self.final_solution_hist_index = 0

    def print_cluster_information(self):
        self.clear_data_information_panel()

        # Gather information
        cluster_centers_label = self.calculate_cluster_centers()
        rounded_cluster_centers = [ (round(center[0], 2), round(center[1], 2)) for center in cluster_centers_label ]
        cluster_items = self.get_cluster_items() # It is a dict like {0:[(x1, y1), (x2, y2)], 1:[(x3, y3), (x4, y4)], 1: ...}
        pair_objectives, max_pair_objective = self.calculate_pair_objectives()

        self.add_data_infromation_panel("Clustering labels: " + str(self.get_cluster_vector()))
        self.add_data_infromation_panel("\nCluster centers: " + str(rounded_cluster_centers).replace("  ", "").replace("\n", " ")+"\n")

        self.add_data_infromation_panel("There are " + str(len(cluster_items)) + " clusters:\n")
        for cluster_id, cluster_items in cluster_items.items():
            self.add_data_infromation_panel("\nCluster " + str(cluster_id) + " items: " + 
                str( [ (round(item.get_coordinates()[0], 2), round(item.get_coordinates()[1], 2)) for item in cluster_items ] ))
            
        self.add_data_infromation_panel("\n\nFarthest Hub Distances: \n" + str(self.calculate_distances_from_center()))
        self.add_data_infromation_panel("\n\nAll Possible Pairs: \n" + str([ (point_tuple[0].get_id(), point_tuple[1].get_id())
                                                                            for point_tuple in self.calculate_all_possible_pairs() ]))
        self.add_data_infromation_panel("\n\nPair Objectives: \n" + str(pair_objectives))
        self.add_data_infromation_panel("\n\nMax Pair Objective: \n" + str(max_pair_objective))
                                                                                              

    def add_data_infromation_panel(self, data):
        old_text = self.monitor_information_panel.toPlainText()
        self.monitor_information_panel.setText(
            old_text + data
        )

    def clear_data_information_panel(self):
        self.monitor_information_panel.clear()

    def add_data_results_panel(self, data):
        old_text = self.monitor_results.toPlainText()
        self.monitor_results.setText(
            old_text + "\n" + data
        )
    
    def clear_data_results_panel(self):
        self.monitor_results.clear()

    # Clustering functions

    def clustering_button_clicked(self):
        sender = self.sender()

        if sender.text() == 'K-Means':
            dialog = Get_Data_Dialog(["Number of clusters: ", ["Init: ", "k-means++", "random"], "Max iterations: ", ["Algorithm: ", "auto", "full", "elkan"]])
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                data = dialog.get_input()
                
                args_dict = { 
                    "n_clusters": int(data[0]) if data[0] != '' else 3, # Default value is 3
                    "init": data[1],
                    "max_iter": int(data[2]) if data[2] != '' else 300, # Default value is 300
                    "algorithm": data[3]
                }
            else:
                return

        elif sender.text() == 'Affinity Propagation':
            dialog = Get_Data_Dialog(
                ["Damping: ", "Max iterations: ", "Convergence iteration: "]
            )
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                data = dialog.get_input()
                
                args_dict = { 
                    "damping": float(data[0]) if data[0] != '' else 0.5, # Default value is 0.5
                    "max_iter": int(data[1]) if data[1] != '' else 200, # Default value is 200
                    "convergence_iter": int(data[2]) if data[2] != '' else 15, # Default value is 15
                }
            else:
                return

        elif sender.text() == 'Mean Shift':
            dialog = Get_Data_Dialog(["Bandwidth: ", "Max iterations: "])
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                data = dialog.get_input()
                
                args_dict = { 
                    "bandwidth": float(data[0]) if data[0] != '' else None, # Default value is None
                    "max_iter": int(data[1]) if data[1] != '' else 300, # Default value is 300
                }
                print("Invalid input. Using default values.")
            else:
                return

        elif sender.text() == 'Spectral Clustering':
            dialog = Get_Data_Dialog(["Number of clusters: ", ["Assign labels: ", "kmeans", "discretize"], "Eigen solver: ", "Random state: "])
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                data = dialog.get_input()
                
                args_dict = { 
                    "n_clusters": int(data[0]) if data[0] != '' else 8, # Default value is 8
                    "assign_labels": data[1],
                    "eigen_solver": data[2] if data[2] != '' else None, # Default value is None
                    "random_state": int(data[3]) if data[3] != '' else None, # Default value is None
                }
                print("Invalid input. Using default values.")
            else:
                return

        elif sender.text() == 'Hierarchical Clustering':
            dialog = Get_Data_Dialog(["Number of clusters: ", ["Linkage: ", "ward", "complete", "average", "single"], "Distance threshold: "])
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                data = dialog.get_input()
                
                args_dict = { 
                    "n_clusters": int(data[0]) if data[0] != '' else 2, # Default value is 2
                    "linkage": data[1],
                    "distance_threshold": float(data[2]) if data[2] != '' else None, # Default value is None
                }
                print("Invalid input. Using default values.")
            else:
                return

        elif sender.text() == 'DBSCAN':
            dialog = Get_Data_Dialog(["Epsilon: ", "Min samples: ", "Metric: "])
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                data = dialog.get_input()
                
                args_dict = { 
                    "eps": float(data[0]) if data[0] != '' else 0.5, # Default value is 0.5
                    "min_samples": int(data[1]) if data[1] != '' else 5, # Default value is 5
                    "metric": data[2] if data[2] != '' else 'euclidean', # Default value is 'euclidean'
                }
                print("Invalid input. Using default values.")
            else:
                return

        self.method_handler_clustering(sender.text(), args_dict)
        self.plot_final_solution()
        self.print_cluster_information()

    # Heuristics functions

    def heuristics_button_clicked(self):
        sender = self.sender()
        self.method_handler_heuristics(sender.text())

    def initial_solution_button_clicked(self):
        sender = self.sender()
        
        # Save and Export operations
        if sender.text() == 'Save As': # Save as txt
            
            point_matrix = self.get_data()
            with open(QtWidgets.QFileDialog.getSaveFileName(self, 'Save As', "initial_solution.txt", "Text files (*.txt)")[0], 'w') as f:
                for point in point_matrix:
                    f.write(str(point.get_coordinates()[0]) + " " + str(point.get_coordinates()[1]) + "\n")
        elif sender.text() == 'Save':
            point_matrix = self.get_data()
            with open("initial_solution.txt", 'w') as f:
                for point in point_matrix:
                    f.write(str(point.get_coordinates()[0]) + " " + str(point.get_coordinates()[1]) + "\n")
        elif sender.text() == 'Export As': # export as jpg
            self.monitor_initial_solution.pixmap().save(QtWidgets.QFileDialog.getSaveFileName(self, 'Export As', "initial_solution.png", "Images (*.png)")[0])
        
        # Undo and Redo operations
        elif sender.text() == 'Undo':
            
            if self.initial_solution_hist_index < len(self.initial_solution_png_hist) - 1:
                print( "Successfull " if self.undo() else "Unsuccessfull", " undo operation.")

                self.initial_solution_hist_index += 1
                self.monitor_initial_solution.setPixmap(self.initial_solution_png_hist[self.initial_solution_hist_index])
        elif sender.text() == 'Redo':
            
            if self.initial_solution_hist_index > 0:
                print( "Successfull " if self.redo() else "Unsuccessfull", " redo operation.")
            
                self.initial_solution_hist_index -= 1
                self.monitor_initial_solution.setPixmap(self.initial_solution_png_hist[self.initial_solution_hist_index])
            

    def final_solution_button_clicked(self):
        sender = self.sender()

        if sender.text() == 'Save As': # Save as txt
            point_matrix = self.get_data()
            with open(QtWidgets.QFileDialog.getSaveFileName(self, 'Save As', "final_solution.txt", "Text files (*.txt)")[0], 'w') as f:
                for point in point_matrix:
                    f.write(str(point.get_coordinates()[0]) + " " + str(point.get_coordinates()[1]) + " cluster: " + str(point.get_cluster_id()) + "\n")
                
        elif sender.text() == 'Save':
            point_matrix = self.get_data()
            with open("final_solution.txt", 'w') as f:
                for point in point_matrix:
                    f.write(str(point.get_coordinates()[0]) + " " + str(point.get_coordinates()[1]) + " cluster: " + str(point.get_cluster_id()) + "\n")
        
        elif sender.text() == 'Export As': # export as jpg
            self.monitor_final_solution.pixmap().save(QtWidgets.QFileDialog.getSaveFileName(self, 'Export As', "final_solution.png", "Images (*.png)")[0])

        elif sender.text() == 'Undo':
            if self.final_solution_hist_index < len(self.final_solution_png_hist) - 1:
                self.final_solution_hist_index += 1
                self.monitor_final_solution.setPixmap(self.final_solution_png_hist[self.final_solution_hist_index])

        elif sender.text() == 'Redo':
            if self.final_solution_hist_index > 0:
                self.final_solution_hist_index -= 1
                self.monitor_final_solution.setPixmap(self.final_solution_png_hist[self.final_solution_hist_index])
                

    ##############################################################

    ###################### Common Operations #####################

    def exit_app(self):
        sys.exit()
       

    def plot_to_pixmap(self, fig, label_size):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, label_size)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return self.cv2_to_pixmap(img)

    def cv2_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qImg)


if __name__ == '__main__':
    print("Testing UI_Script class")

    app = QtWidgets.QApplication(sys.argv)
    window = UI_Interface(template_path='Interface.ui')
    sys.exit(app.exec_())
