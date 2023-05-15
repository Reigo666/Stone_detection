from client import window
from PyQt5.QtWidgets import QApplication
import sys
from PyQt5 import QtCore


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    w = window.MainWindow()
    w.show()
    sys.exit(app.exec_())
