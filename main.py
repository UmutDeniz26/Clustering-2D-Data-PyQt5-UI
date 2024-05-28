
import sys
from PyQt5 import QtWidgets
from UI_script import PyqtUI


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = PyqtUI(template_path='Interface.ui')
    sys.exit(app.exec_())