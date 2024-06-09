
import sys
from PyQt5 import QtWidgets
from UI_Interface import UI_Interface

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = UI_Interface(template_path='Interface.ui')
    sys.exit(app.exec_())