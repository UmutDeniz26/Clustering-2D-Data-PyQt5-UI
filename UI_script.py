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

# Clustering_Operations requires Point_Matrix class from Data.py from init
class PyqtUI(QMainWindow, Clustering_Operations):
    def __init__(self, template_path):
        Clustering_Operations.__init__(self, data=None)  # Pass 'data' argument
        QMainWindow.__init__(self , data=None)  # Pass 'data' argument


        uic.loadUi(template_path, self)
        self.show()
        
        # Load qrc file
        # qrc path = src_side/resource.qrc
        # qrc file = resource.qrc
        

        self.icon_only_widget.setVisible(True)
        self.full_menu_widget.setVisible(False)
        self.sidebar_button.clicked.connect(self.sidebar_button_clicked)
        self.open_data.clicked.connect(self.load_data_button)

        self.clustering_kmeans.clicked.connect(self.buttonClicked)
        self.clustering_affinity_propagation.clicked.connect(self.buttonClicked)
        self.clustering_mean_shift.clicked.connect(self.buttonClicked)

        """
        ###################### File Operations ######################
        # Source operations
        self.source_folder.clicked.connect(self.load_image_button);self.source_folder_menu.triggered.connect(self.source_folder.click)
        self.source_export.clicked.connect(self.export_source_image);self.source_export_menu.triggered.connect(self.source_export.click)

        # Output operations
        self.output_save.clicked.connect(self.save_output_image);self.output_save_menu.triggered.connect(self.output_save.click)
        self.output_save_as.clicked.connect(self.save_as_output_image);self.output_save_as_menu.triggered.connect(self.output_save_as.click)
        self.output_export.clicked.connect(self.export_output_image);self.output_export_menu.triggered.connect(self.output_export.click)
        
        ###################### Common Operations #####################
        self.output_undo.clicked.connect(self.undo_output_image);self.output_undo_menu.triggered.connect(self.output_undo.click)
        self.output_redo.clicked.connect(self.redo_output_image);self.output_redo_menu.triggered.connect(self.output_redo.click)
        self.exit_menu.triggered.connect(self.exit_app)                
        self.source_clear_menu.triggered.connect(self.clear_source_image)
        self.output_clear_menu.triggered.connect(self.clear_output_image)


        self.admin_print_menu.triggered.connect(self.admin_print)

        # Button lists
        self.all_buttons = [
            self.source_folder, self.source_folder_menu, self.source_export_menu, self.source_clear_menu, self.source_export, self.source_undo
                    
            
            ,self.output_save, self.output_save_as,self.output_undo_menu,self.output_save_menu, self.output_save_as_menu, self.output_export_menu,self.output_clear_menu,self.output_redo_menu
            ,self.output_export, self.output_undo, self.output_redo
            
            ,self.bgr_2_gray, self.bgr_2_hsv

            ,self.segment_multi_otsu, self.segment_chan_vese, self.segment_moprh_snakes

            ,self.edge_roberts, self.edge_sobel, self.edge_scharr, self.edge_prewitt

            ,self.exit_menu
        ]

        self.menu_buttons = [
            self.source_folder_menu, self.source_export_menu, self.source_clear_menu, 
            self.output_save_menu, self.output_save_as_menu, self.output_export_menu, self.output_clear_menu, self.output_undo_menu, self.output_redo_menu
        ]


        # Always display buttons
        self.always_display_buttons = [
            self.source_folder,self.source_folder_menu, self.exit_menu
        ]
        
        # Disable the buttons
        self.change_button_state(self.all_buttons, False)
        """
    
    
    #############################################################

    ###################### File Operations ######################

    def load_data_button(self):
        data_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "data.txt", "Data files (*.txt)")[0]
        if data_path:
            self.set_initial_solution(data_path)
            self.update_initial_solution()
    
    
    def sidebar_button_clicked(self):
        self.full_menu_widget.setVisible(not self.full_menu_widget.isVisible())
        self.icon_only_widget.setVisible(not self.icon_only_widget.isVisible())











    ##############################################################

    ###################### Data Operations ######################


    # Display functions
    def update_initial_solution(self, label_size=(400, 400)):
        # get plot
        initial_solution_data = self.get_initial_solution().get_data_as_list()
        
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

    def update_final_solution(self, label_size=(400, 400)):
        # get plot
        final_solution_data = self.get_final_solution().get_data_as_list()
        
        # 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        for point in final_solution_data:
            ax.scatter(point[0], point[1])
        
        # Convert plot to pixmap
        pixmap = self.plot_to_pixmap(fig, label_size)
        self.monitor_final_solution.setPixmap(pixmap)
        plt.close(fig)

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
        
    # Clustering functions

    def buttonClicked(self):
        sender = self.sender()
        print(sender.text())
        self.method_handler(sender.text())

    # Clear functions
    def clear_source_image(self):
        
        self.image_operator.set_source_image(None);self.source_image_path = None
        self.image_operator.source_image_history = {"image_history":[], "current_index":-1}

        self.change_button_state(self.all_buttons, False)
        self.source_image.clear()

    def clear_output_image(self):
        self.image_operator.output_image_history = {"image_history":[], "current_index":-1}
        self.output_image.clear()


    # Edge detection functions



    ##############################################################

    ###################### Common Operations #####################

    def exit_app(self):
        sys.exit()
       
    def change_button_state(self, button, state):
        for button in self.all_buttons:
            if button not in self.always_display_buttons:
                button.setDisabled(False) if state else button.setDisabled(True)

    def undo_output_image(self):
        self.image_operator.undo_output_image()
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def redo_output_image(self):
        self.image_operator.redo_output_image()
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))


if __name__ == '__main__':
    print("Testing UI_Script class")

    app = QtWidgets.QApplication(sys.argv)
    window = PyqtUI(template_path='Interface.ui')
    sys.exit(app.exec_())
