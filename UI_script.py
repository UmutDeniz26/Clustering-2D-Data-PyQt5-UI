# Read PyqtUI.ui file and show the window
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
# Clustering_Operations requires Point_Matrix class from Data.py from init
class PyqtUI(QMainWindow, Clustering_Operations, Heuristic_Operations):
    def __init__(self, template_path):
        super(PyqtUI, self).__init__( data = None )

        uic.loadUi(template_path, self)
        self.show()
        
        self.full_menu_widget.setVisible(False)
        self.open_data.clicked.connect(self.load_data_button)

        self.side_menu_buttons = [ self.initial_solution_side, self.final_solution_side, self.clustering_side, self.heuristics_side ]
        for button in self.side_menu_buttons:
            button.clicked.connect(self.sidebar_button_clicked)

        self.always_display_buttons = [self.open_data, self.exit_button, self.menu_open_data, self.menu_exit, self.menu_file]

        self.disable_all_buttons()

        print("UI_Script class initialized.")
        print("All buttons: ", self.get_buttons())



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
    
    def disable_all_buttons(self):
        # Get all buttons except the always display buttons
        for button in self.get_buttons():
            if button in self.always_display_buttons:
                continue
            button.setDisabled(True)


    def enable_buttons(self):
        for button in self.get_buttons():
            button.setDisabled(False)
            
    
    
    #############################################################

    ###################### File Operations ######################

    def load_data_button(self):
        data_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "data.txt", "Data files (*.txt)")[0]
        if data_path:
            self.load_data(data_path)
            self.plot_initial_solution()
            self.enable_buttons()
    
    

    ###################### Side Bar ######################
    
    def sidebar_button_clicked(self):
        sender = self.sender()

        if not hasattr(self, 'hold_sender') or self.hold_sender != sender or not self.full_menu_widget.isVisible():
            self.full_menu_widget.setVisible(True)
        else:
            self.full_menu_widget.setVisible(False)

        self.edit_full_menu_buttons(sender, self.toolbox_layout)    
        
        self.hold_sender = sender

    def edit_full_menu_buttons(self, sender, container):
        # Clear buttons of the container
        for i in reversed(range(container.count())):
            container.itemAt(i).widget().deleteLater()
        
        if sender == self.initial_solution_side:
            # Buttons to add
            button_names = ['Save As', 'Save', 'Export As', 'Undo', 'Redo']
            button_object_names = ['side_initial_save_as', 'side_initial_save', 'side_initial_export_as', 'side_initial_undo', 'side_initial_redo']

            # Add buttons to the container
            for i,button_name in enumerate(button_names):
                button = QtWidgets.QPushButton(button_name)
                button.setObjectName(button_object_names[i])

                button.clicked.connect(self.initial_solution_button_clicked)
                container.addWidget(button)

        elif sender == self.final_solution_side:
            # Buttons to add
            button_names = ['Save As', 'Save', 'Export As', 'Undo', 'Redo']
            button_object_names = ['side_final_save_as', 'side_final_save', 'side_final_export_as', 'side_final_undo', 'side_final_redo']

            # Add buttons to the container
            for i,button_name in enumerate(button_names):
                button = QtWidgets.QPushButton(button_name)
                button.setObjectName(button_object_names[i])

                button.clicked.connect(self.final_solution_button_clicked)
                container.addWidget(button)

        elif sender == self.clustering_side:
            # Buttons to add
            button_names = ['K-Means', 'Affinity Propagation', 'Mean Shift', 'Spectral Clustering', 'Hierarchical Clustering', 'DBSCAN']

            # Add buttons to the container
            for button_name in button_names:
                button = QtWidgets.QPushButton(button_name)
                button.clicked.connect(self.clustering_button_clicked)
                container.addWidget(button)

        elif sender == self.heuristics_side:
            # Buttons to add
            button_names = ['Hill Climbing', 'Simulated Annealing']

            # Add buttons to the container
            for button_name in button_names:
                button = QtWidgets.QPushButton(button_name)
                button.clicked.connect(self.heuristics_button_clicked)
                container.addWidget(button)

            

        
        
        







    ##############################################################

    ###################### Data Operations ######################


    # Display functions
    def plot_initial_solution(self, label_size=(400, 400)):
        # get plot
        initial_solution_data = self.get_data_as_list()
        
        # 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        for point in initial_solution_data:
            ax.scatter(point[0], point[1], c='black')
        
        # Convert plot to pixmap
        pixmap = self.plot_to_pixmap(fig, label_size)
        self.monitor_initial_solution.setPixmap(pixmap)
        plt.close(fig)

    def plot_final_solution(self, label_size=(400, 400)):
        if len(self.get_cluster_vector()) == 0:
            raise ValueError("Cluster vector is empty.")
        else:
            cluster_vector = self.get_cluster_vector()
            cluster_id_vector = self.get_cluster_id_vector()
            data = self.get_data_as_list()

            # 2D plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')

            color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown']

            for i in range(len(data)):
                ax.scatter(data[i][0], data[i][1], c=color_list[cluster_vector[i]])
            
            try:
                for cluster_id in cluster_id_vector:
                    ax.scatter(cluster_id[0], cluster_id[1], c='red', s=100, marker='x')
            except:
                print("Cluster id vector is empty.")

            # Convert plot to pixmap
            pixmap = self.plot_to_pixmap(fig, label_size)
            self.monitor_final_solution.setPixmap(pixmap)
            plt.close(fig)

        
    # Clustering functions

    def clustering_button_clicked(self):
        sender = self.sender()
        print(sender.text())
        self.method_handler_clustering(sender.text())
        self.plot_final_solution()

    # Heuristics functions

    def heuristics_button_clicked(self):
        sender = self.sender()
        self.method_handler_heuristics(sender.text())

    def initial_solution_button_clicked(self):
        sender = self.sender()
        print(sender.text())        

    def final_solution_button_clicked(self):
        sender = self.sender()
        print(sender.text())


    ##############################################################

    ###################### Common Operations #####################

    def exit_app(self):
        sys.exit()
       
    def change_button_state(self, button, state):
        for button in self.all_buttons:
            if button not in self.always_display_buttons:
                button.setDisabled(False) if state else button.setDisabled(True)

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
    window = PyqtUI(template_path='Interface.ui')
    sys.exit(app.exec_())
