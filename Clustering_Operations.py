import numpy as np
import cv2
import skimage as ski
from Data import Point_Matrix

class Clustering_Operations( Point_Matrix ):
    def __init__(self, data: Point_Matrix):
        """
        Constructor for image_operator class
        :param image: np.ndarray or str
        """
        super().__init__("data.txt", data)

    def set_data(self, data: Point_Matrix):
        if isinstance(data, str):
            self.data = cv2.imread(data, cv2.IMREAD_COLOR)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError('Invalid input, type: ', type(data))


if __name__ == '__main__':
    pass
