
import sys
from PyQt5 import QtWidgets, QtGui


class Get_Data_Dialog(QtWidgets.QDialog):
    def __init__(self, questions = ["Default question:"]):
        super(Get_Data_Dialog, self).__init__()
        self.setWindowTitle("Get Data Dialog")
        self.setGeometry(100, 100, len(questions)*200, 100)
        
        self.questions = questions
        self.labels = []
        self.inputs = []
        self.init_ui()
        

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout()

        for question in self.questions:
            
            if isinstance(question, list):
                # ComboBox after 0th element
                label = QtWidgets.QLabel(question[0])
                self.labels.append(label)
                self.layout.addWidget(label)

                combobox = QtWidgets.QComboBox()
                for item in question[1:]:
                    combobox.addItem(item)
                self.inputs.append(combobox)
                self.layout.addWidget(combobox)    
            else:
                label = QtWidgets.QLabel(question)
                self.labels.append(label)
                self.layout.addWidget(label)

                textbox = QtWidgets.QLineEdit()
                self.inputs.append(textbox)
                self.layout.addWidget(textbox)

        self.buttonbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.layout.addWidget(self.buttonbox)

        self.setLayout(self.layout)

    def get_input(self):
        inputs = []
        for textbox in self.inputs:
            if isinstance(textbox, QtWidgets.QLineEdit):
                inputs.append(textbox.text())
            elif isinstance(textbox, QtWidgets.QComboBox):
                inputs.append(textbox.currentText())
        return inputs

    def __del__(self):
        print("Get_Data_Dialog deleted.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = Get_Data_Dialog(["What is your name?", "What is your age?", ["What is your favorite color?", "Red", "Green", "Blue"]])
    dialog.exec_()
    print(dialog.get_input())
    sys.exit(app.exec_())
    