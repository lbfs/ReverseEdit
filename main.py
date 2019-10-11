#!/usr/bin/env python3
import sys
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QTableWidget, QHeaderView
from PySide2.QtCore import QFile, QObject, QDir

class ReverseEditWindow(QWidget):
    def __init__(self, parent=None):
        super(ReverseEditWindow, self).__init__(parent)
        ui_file = QFile("mainwindow.ui")
        ui_file.open(QFile.ReadOnly)

        loader = QUiLoader()
        self.window = loader.load(ui_file)

        # SourceVideoList
        headers = ["Source Filename"]
        self.source_video_list = self.window.findChild(QTableWidget, "SourceVideoList")
        self.source_video_list.setColumnCount(1)
        self.source_video_list.setHorizontalHeaderLabels(headers)
        self.source_video_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # AddVideoButton
        self.add_video_button = self.window.findChild(QPushButton, "AddVideoButton")
        self.add_video_button.clicked.connect(self.add_video)

        # RemoveVideoButton
        self.remove_video_button = self.window.findChild(QPushButton, "RemoveVideoButton")
        self.remove_video_button.clicked.connect(self.remove_video)

        self.window.show()

    def add_video(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Video Types (*.mp4 *.mkv *.webm)")
        data = dialog.getOpenFileNames(self, "Select Source Video", QDir.homePath(), "Video Types (*.mp4 *.mkv *.webm)")
        filenames = data[0]
        for filename in filenames:
            print(filename)

    def remove_video(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ReverseEditWindow()
    sys.exit(app.exec_())
