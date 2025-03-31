# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'overwrite_type.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QGridLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton,
    QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(1121, 192)
        self.gridLayout_3 = QGridLayout(Dialog)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.widget = QWidget(Dialog)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.filePath = QLineEdit(self.widget)
        self.filePath.setObjectName(u"filePath")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(10)
        self.filePath.setFont(font)

        self.gridLayout.addWidget(self.filePath, 0, 1, 1, 1)

        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.selectFileButton = QPushButton(self.widget)
        self.selectFileButton.setObjectName(u"selectFileButton")
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(11)
        self.selectFileButton.setFont(font1)

        self.gridLayout.addWidget(self.selectFileButton, 0, 2, 1, 1)


        self.gridLayout_3.addWidget(self.widget, 0, 0, 1, 1)

        self.edfTypeWig = QWidget(Dialog)
        self.edfTypeWig.setObjectName(u"edfTypeWig")
        self.gridLayout_2 = QGridLayout(self.edfTypeWig)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_2 = QLabel(self.edfTypeWig)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)

        self.edfD = QRadioButton(self.edfTypeWig)
        self.edfTypeButtonGroup = QButtonGroup(Dialog)
        self.edfTypeButtonGroup.setObjectName(u"edfTypeButtonGroup")
        self.edfTypeButtonGroup.addButton(self.edfD)
        self.edfD.setObjectName(u"edfD")
        self.edfD.setAutoExclusive(False)

        self.gridLayout_2.addWidget(self.edfD, 0, 1, 1, 1)

        self.edfC = QRadioButton(self.edfTypeWig)
        self.edfTypeButtonGroup.addButton(self.edfC)
        self.edfC.setObjectName(u"edfC")
        self.edfC.setAutoExclusive(False)

        self.gridLayout_2.addWidget(self.edfC, 0, 2, 1, 1)


        self.gridLayout_3.addWidget(self.edfTypeWig, 1, 0, 1, 1, Qt.AlignHCenter)

        self.convertButton = QPushButton(Dialog)
        self.convertButton.setObjectName(u"convertButton")
        self.convertButton.setFont(font1)

        self.gridLayout_3.addWidget(self.convertButton, 2, 0, 1, 1, Qt.AlignHCenter)


        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"Input file", None))
        self.selectFileButton.setText(QCoreApplication.translate("Dialog", u"Select file...", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"Convert to EDF type:", None))
        self.edfD.setText(QCoreApplication.translate("Dialog", u"EDF+D", None))
        self.edfC.setText(QCoreApplication.translate("Dialog", u"EDF+C", None))
        self.convertButton.setText(QCoreApplication.translate("Dialog", u"Convert", None))
    # retranslateUi

