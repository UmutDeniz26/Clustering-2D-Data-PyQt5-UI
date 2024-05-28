import numpy as np
import cv2
import skimage as ski

class Partitioning_Algorithms:
    def __init__(self, data:np.ndarray = None):
        """
        Constructor for image_operator class
        :param image: np.ndarray or str
        """

        if data is not None:
            self.set_data( data )

    def set_data(self, data: np.ndarray):
        if isinstance(data, str):
            self.data = cv2.imread(data, cv2.IMREAD_COLOR)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError('Invalid input, type: ', type(data))
        
       
if __name__ == '__main__':
    pass
