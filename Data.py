class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def get_coordinates(self):
        return self.x, self.y, self.z
    
    def set_coordinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Point_Cloud:
    def __init__(self, filename, data=None):
        
        self.filename = filename

        self.data = data if data else []

    def load_data(self):
        try:
            with open(self.filename, 'r') as file:
                txt_data = file.read()
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("File not found.")

        txt_data = txt_data.split("\n")
        for line in txt_data:
            if line:
                x, y, z = line.split(" ")
                self.data.append(Point(float(x), float(y), float(z)))

    def set_data(self, data):
        for point in data:
            if not isinstance(point, Point):
                raise ValueError("Invalid input, type: ", type(point))
            self.data.append(point)
            
        
    def clear_data(self):
        self.data = []

    def get_data(self):
        return self.data

    def save_data(self, filename, solution):
        try:
            with open(filename, 'w') as file:
                file.write(solution)
            print("Data saved successfully.")
        except:
            print("Error occurred while saving data.")

# Example usage:
# data = Data("example.txt")
# data.load_data()
# data.save_data("solution.txt", "This is a solution.")
