class Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def get_coordinates(self):
        return self.x, self.y, self.z
    
    def set_coordinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Point_Matrix:
    def __init__(self, filename= None, data=None):
        self.set_filename(filename)
        self.set_data(data)

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
                        x, y, z = line.split(" ")
                        self.data.append(Point(float(x), float(y), float(z)))

                print("Data loaded successfully: ")
                self.print_data()
        except FileNotFoundError:
            print("File not found.")


    def set_data(self, data):
        print(data)
        if isinstance(data, list):
            for point in data:
                print(point)
                self.data.append(Point(point[0], point[1], point[2]))
        
        elif isinstance(data, Point_Matrix):
            self.data = data.get_data()
        
        elif isinstance(data, str):
            self.load_data(data)

        elif data is None:
            self.clear_data()

    def get_data(self):
        return self.data

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
