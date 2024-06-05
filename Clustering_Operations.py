import numpy as np
import cv2
import skimage as ski
from Data import Point_Matrix

from sklearn import cluster


class Clustering_Operations( ):
    def __init__(self, data: Point_Matrix):
        """
        Constructor for image_operator class
        :param image: np.ndarray or str
        """
        self.set_initial_solution(data)
        
    def set_initial_solution(self, data: Point_Matrix):
        if isinstance(data, Point_Matrix):
            self.initial_solution = data
        elif isinstance(data, str):
            self.initial_solution = Point_Matrix(data = data)
        elif data is None:
            self.initial_solution = Point_Matrix()

    def get_initial_solution(self):
        return self.initial_solution
    
    def set_final_solution(self, data: Point_Matrix):
        if isinstance(data, Point_Matrix):
            self.final_solution = data
        elif isinstance(data, str):
            self.final_solution = Point_Matrix(data = data)
        elif data is None:
            self.final_solution = Point_Matrix()

    def get_final_solution(self):
        return self.final_solution

    def method_handler(self,method_name):
        if method_name == 'K-Means':
            self.kmeans()
        elif method_name == 'clustering_gmm':
            self.gmm()
        else:
            print("Method not found.")

    def kmeans(self):
        # Get data
        data = self.get_initial_solution().get_data_as_list()
        data = np.array(data)
        
        # Get number of clusters
        n_clusters = 3
        
        # KMeans
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        
        # Get cluster vector
        cluster_vector = kmeans.predict(data)
        print(cluster_vector)
        print(kmeans.cluster_centers_)
        print(kmeans.labels_)

if __name__ == '__main__':
    pass
