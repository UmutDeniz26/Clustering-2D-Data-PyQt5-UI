
# Class to represent a point in 3D space
class Point:
    def __init__(self, x, y,cluster_id=None):
        self.__x = float(x)
        self.__y = float(y)
        self.set_cluster_id(cluster_id)

    def __str__(self):
        return f"({self.__x}, {self.__y}), Cluster: {self.get_cluster_id()}\n"

    def set_cluster_id(self, cluster_id):
        self.__cluster_id = cluster_id

    def get_cluster_id(self):
        return self.__cluster_id
    
    def set_coordinates(self, x, y):
        self.__x = x
        self.__y = y
    
    def get_coordinates(self):
        return self.__x, self.__y

# Class to represent a matrix of points
class Point_Matrix:
    def __init__(self, filename= None, data=None):
        self.set_filename(filename)
        self.set_data(data)
        self.set_cluster_vector([])
        self.set_cluster_id_vector([])

    def load_data(self, filename=None):

        # Set filename if provided
        if filename is not None:
            self.set_filename(filename)

        # Clear data
        self.clear_data()
        
        try:
            # Read data from file
            with open(self.get_filename(), 'r') as file:
                data = file.read()

                data = data.split("\n")
                for line in data:
                    if line:
                        x, y = line.split(" ")
                        self.data.append(Point(float(x), float(y)))

                print("Data loaded successfully: ")
                self.print_data()
        except FileNotFoundError:
            print("File not found.")

    
    def set_cluster_vector(self, cluster_vector):
        self.cluster_vector = cluster_vector

    def get_cluster_vector(self):
        return self.cluster_vector
    
    def set_cluster_id_vector(self, cluster_id_vector):
        self.cluster_id_vector = cluster_id_vector

    def get_cluster_id_vector(self):
        return self.cluster_id_vector
    

    def set_data(self, data):
        if isinstance(data, list):
            for point in data:
                print(point)
                self.data.append(Point(point[0], point[1]))
        
        elif isinstance(data, Point_Matrix):
            self.data = data.get_data()
        
        elif isinstance(data, str):
            self.load_data(data)

        elif data is None:
            self.clear_data()

    def get_data(self):
        return self.data
    
    def get_data_as_list(self):
        data = []
        for point in self.get_data():
            data.append(point.get_coordinates())
        return data

    def clear_data(self):
        self.data = []

    def print_data(self):
        for point in self.get_data():
            print(point)

    def set_filename(self, filename):
        self.filename = filename

    def get_filename(self):
        return self.filename

    def save_data(self, solution, filename=None):
        
        # Set filename if provided
        if filename is not None:
            self.set_filename(filename) 

        try:
            with open(self.get_filename(), 'w') as file:
                file.write(solution)
            print("Data saved successfully.")
        except:
            print("Error occurred while saving data.")

if __name__ == '__main__':
    pcd = Point_Matrix()
    pcd.load_data("src/points.txt")
