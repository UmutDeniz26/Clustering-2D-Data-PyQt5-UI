# Read PyqtUI.ui file and show the window
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import sys
import cv2
import numpy as np
import os
from Partitioning_Algorithms import Partitioning_Algorithms

class PyqtUI(QMainWindow):
    def __init__(self, template_path):
        super().__init__()
        uic.loadUi(template_path, self)
        self.show()

        self.partitioning_algorithms = Partitioning_Algorithms()
        
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
        


    def admin_print(self):
        print("Histories. \nSource")
        print(self.image_operator.source_image_history)

        print("Output")
        print(self.image_operator.output_image_history)






    #############################################################

    ###################### File Operations ######################

    def load_image_button(self):
        
        # Get the image path
        image_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png)")[0]
        
        # If the image path is not empty, set the source image
        if image_path:
            # Name of the label of the source image is sourceImage
            self.image_operator.set_source_image(image_path)            
            self.update_source_image(label_size=(self.source_image.width(), self.source_image.height()))

            # Assign the source image path
            self.source_image_path = image_path

            # Enable the buttons
            self.change_button_state(self.all_buttons, True)
    
    def save_output_image(self):
        # If the folder path is not empty, save the output image
        if self.source_image_path:
            self.image_operator.get_output_image().save_image(self.source_image_path)

    def save_as_output_image(self):

        while True:
            # Get the folder path
            image_save_path = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save file', "output.jpg", "Image files (*.jpg)")[0]

            if not image_save_path:
                break

            # If the folder path is not empty, save the output image, check extension is jpg
            if image_save_path.endswith('.jpg'):
                self.image_operator.get_output_image().save_image(image_save_path)
                break
            else:
                print("Please select a valid path with .jpg extension")

    def export_output_image(self):
        while True:
            # Custom extension choice (jpg, png, bmp)
            image_save_path = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save file', "output.jpg", "Image files (*.jpg *.png *.bmp)")[0]
            
            if not image_save_path:
                break

            # If the folder path is not empty, save the output image, check extension is jpg or png or bmp
            if (image_save_path.endswith('.jpg') or image_save_path.endswith('.png') or image_save_path.endswith('.bmp')):
                self.image_operator.get_output_image().save_image(image_save_path)
                break
            else:
                print("Please select a valid path with .jpg or .png or .bmp extension : ", image_save_path)



    def export_source_image(self):
        while True:
            # Custom extension choice (jpg, png, bmp)
            image_save_path = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save file', "source.jpg", "Image files (*.jpg *.png *.bmp)")[0]
            
            # If the folder path is not empty, save the output image
            if not image_save_path:
                break

            # If the folder path is not empty, save the output image, check extension is jpg or png or bmp
            if (image_save_path.endswith('.jpg') or image_save_path.endswith('.png') or image_save_path.endswith('.bmp')):
                self.image_operator.get_source_image().save_image(image_save_path)
                break
            else:
                print("Please select a valid path with .jpg or .png or .bmp extension : ", image_save_path)













    ##############################################################

    ###################### Image Operations ######################


    # Display functions
    def update_source_image(self, label_size=(400, 400)):
        self.source_image.setPixmap(
            QPixmap.fromImage(
                self.image_operator.get_source_image().get_QImage()
            ).scaled(label_size[0], label_size[1])
        )
        self.source_image.setAlignment(Qt.AlignCenter)
    
    def update_output_image(self, label_size=(400, 400)):
        try:
            self.output_image.setPixmap(
                QPixmap.fromImage(
                    self.image_operator.get_output_image().get_QImage()
                ).scaled(label_size[0], label_size[1])
            )
            self.output_image.setAlignment(Qt.AlignCenter)
        except:
            if len (self.image_operator.get_output_image().get_nd_image().shape) < 2:
                # it means img is numpy array that comes from segmentation
                self.output_image.setPixmap(
                    QPixmap.fromImage(
                        self.image_operator.get_output_image().get_nd_image()
                    ).scaled(label_size[0], label_size[1])
                )
                self.output_image.setAlignment(Qt.AlignCenter)
            else:
                print("Error in update_output_image, image shape: ", self.image_operator.get_output_image().get_nd_image().shape)
            

    # Conversion functions
    def convert_to_gray(self):
        img = self.image_operator.convesion_actions(method='gray')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def convert_to_hsv(self):
        img = self.image_operator.convesion_actions(method='hsv')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))


    # Clear functions
    def clear_source_image(self):
        
        self.image_operator.set_source_image(None);self.source_image_path = None
        self.image_operator.source_image_history = {"image_history":[], "current_index":-1}

        self.change_button_state(self.all_buttons, False)
        self.source_image.clear()

    def clear_output_image(self):
        self.image_operator.output_image_history = {"image_history":[], "current_index":-1}
        self.output_image.clear()


    # Segmentation functions
    def segment_multi_otsu_f(self):
        img = self.image_operator.segment_image(method='multi_otsu')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def segment_chan_vese_f(self):
        img = self.image_operator.segment_image(method='chan_vese')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def segment_moprh_snakes_f(self):
        img = self.image_operator.segment_image(method='morph_snakes')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))


    # Edge detection functions
    def edge_roberts_f(self):
        img = self.image_operator.edge_detection_actions(method='roberts')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def edge_sobel_f(self):
        img = self.image_operator.edge_detection_actions(method='sobel')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def edge_scharr_f(self):
        img = self.image_operator.edge_detection_actions(method='scharr')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))

    def edge_prewitt_f(self):
        img = self.image_operator.edge_detection_actions(method='prewitt')
        self.image_operator.set_output_image(img)
        self.update_output_image(label_size=(self.output_image.width(), self.output_image.height()))






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
