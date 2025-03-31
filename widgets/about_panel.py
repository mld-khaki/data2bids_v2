# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'about_panel.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QGridLayout,
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QTextBrowser, QTextEdit, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(593, 421)
        self.gridLayout_2 = QGridLayout(Dialog)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.widget = QWidget(Dialog)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.versionDate = QLabel(self.widget)
        self.versionDate.setObjectName(u"versionDate")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.versionDate.sizePolicy().hasHeightForWidth())
        self.versionDate.setSizePolicy(sizePolicy)
        self.versionDate.setMinimumSize(QSize(130, 26))
        self.versionDate.setMaximumSize(QSize(16777215, 26))
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(11)
        font.setBold(True)
        self.versionDate.setFont(font)
        self.versionDate.setStyleSheet(u"")
        self.versionDate.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.versionDate, 0, 0, 1, 1)

        self.versionDateEdit = QLabel(self.widget)
        self.versionDateEdit.setObjectName(u"versionDateEdit")
        sizePolicy.setHeightForWidth(self.versionDateEdit.sizePolicy().hasHeightForWidth())
        self.versionDateEdit.setSizePolicy(sizePolicy)
        self.versionDateEdit.setMinimumSize(QSize(0, 26))
        self.versionDateEdit.setMaximumSize(QSize(16777215, 26))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(11)
        self.versionDateEdit.setFont(font1)
        self.versionDateEdit.setStyleSheet(u"margin-left:1px;")

        self.gridLayout.addWidget(self.versionDateEdit, 0, 1, 1, 1)

        self.googleDrive = QLabel(self.widget)
        self.googleDrive.setObjectName(u"googleDrive")
        sizePolicy.setHeightForWidth(self.googleDrive.sizePolicy().hasHeightForWidth())
        self.googleDrive.setSizePolicy(sizePolicy)
        self.googleDrive.setMinimumSize(QSize(130, 26))
        self.googleDrive.setMaximumSize(QSize(16777215, 26))
        self.googleDrive.setFont(font)
        self.googleDrive.setStyleSheet(u"")
        self.googleDrive.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.googleDrive, 1, 0, 1, 1)

        self.googleDriveLink = QTextBrowser(self.widget)
        self.googleDriveLink.setObjectName(u"googleDriveLink")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.googleDriveLink.sizePolicy().hasHeightForWidth())
        self.googleDriveLink.setSizePolicy(sizePolicy1)
        self.googleDriveLink.setMinimumSize(QSize(0, 26))
        self.googleDriveLink.setMaximumSize(QSize(16777215, 26))
        palette = QPalette()
        brush = QBrush(QColor(0, 0, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(239, 239, 239, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        brush2 = QBrush(QColor(255, 255, 255, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Light, brush2)
        brush3 = QBrush(QColor(247, 247, 247, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Midlight, brush3)
        brush4 = QBrush(QColor(119, 119, 119, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Dark, brush4)
        brush5 = QBrush(QColor(159, 159, 159, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush2)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette.setBrush(QPalette.Active, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Active, QPalette.AlternateBase, brush3)
        brush6 = QBrush(QColor(255, 255, 220, 255))
        brush6.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Active, QPalette.ToolTipText, brush)
        brush7 = QBrush(QColor(0, 0, 0, 128))
        brush7.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush7)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Light, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Inactive, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Inactive, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette.setBrush(QPalette.Inactive, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush3)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush7)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Light, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Disabled, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush7)
#endif
        self.googleDriveLink.setPalette(palette)
        self.googleDriveLink.setFont(font1)
        self.googleDriveLink.setAutoFillBackground(False)
        self.googleDriveLink.setStyleSheet(u"")
        self.googleDriveLink.setFrameShape(QFrame.NoFrame)
        self.googleDriveLink.setFrameShadow(QFrame.Plain)
        self.googleDriveLink.setLineWidth(0)
        self.googleDriveLink.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.googleDriveLink.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.googleDriveLink.setLineWrapMode(QTextEdit.NoWrap)
        self.googleDriveLink.setOpenExternalLinks(True)

        self.gridLayout.addWidget(self.googleDriveLink, 1, 1, 1, 1)

        self.documentation = QLabel(self.widget)
        self.documentation.setObjectName(u"documentation")
        sizePolicy.setHeightForWidth(self.documentation.sizePolicy().hasHeightForWidth())
        self.documentation.setSizePolicy(sizePolicy)
        self.documentation.setMinimumSize(QSize(200, 26))
        self.documentation.setMaximumSize(QSize(16777215, 26))
        self.documentation.setFont(font)
        self.documentation.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.documentation, 2, 0, 1, 1)

        self.documentationLink = QTextBrowser(self.widget)
        self.documentationLink.setObjectName(u"documentationLink")
        sizePolicy1.setHeightForWidth(self.documentationLink.sizePolicy().hasHeightForWidth())
        self.documentationLink.setSizePolicy(sizePolicy1)
        self.documentationLink.setMinimumSize(QSize(300, 26))
        self.documentationLink.setMaximumSize(QSize(16777215, 26))
        palette1 = QPalette()
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Light, brush2)
        palette1.setBrush(QPalette.Active, QPalette.Midlight, brush3)
        palette1.setBrush(QPalette.Active, QPalette.Dark, brush4)
        palette1.setBrush(QPalette.Active, QPalette.Mid, brush5)
        palette1.setBrush(QPalette.Active, QPalette.Text, brush)
        palette1.setBrush(QPalette.Active, QPalette.BrightText, brush2)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush2)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Shadow, brush)
        palette1.setBrush(QPalette.Active, QPalette.AlternateBase, brush3)
        palette1.setBrush(QPalette.Active, QPalette.ToolTipBase, brush6)
        palette1.setBrush(QPalette.Active, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Active, QPalette.PlaceholderText, brush7)
#endif
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Light, brush2)
        palette1.setBrush(QPalette.Inactive, QPalette.Midlight, brush3)
        palette1.setBrush(QPalette.Inactive, QPalette.Dark, brush4)
        palette1.setBrush(QPalette.Inactive, QPalette.Mid, brush5)
        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.BrightText, brush2)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush2)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Shadow, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush3)
        palette1.setBrush(QPalette.Inactive, QPalette.ToolTipBase, brush6)
        palette1.setBrush(QPalette.Inactive, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush7)
#endif
        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush4)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Light, brush2)
        palette1.setBrush(QPalette.Disabled, QPalette.Midlight, brush3)
        palette1.setBrush(QPalette.Disabled, QPalette.Dark, brush4)
        palette1.setBrush(QPalette.Disabled, QPalette.Mid, brush5)
        palette1.setBrush(QPalette.Disabled, QPalette.Text, brush4)
        palette1.setBrush(QPalette.Disabled, QPalette.BrightText, brush2)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush4)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Shadow, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.ToolTipBase, brush6)
        palette1.setBrush(QPalette.Disabled, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush7)
#endif
        self.documentationLink.setPalette(palette1)
        self.documentationLink.setFont(font1)
        self.documentationLink.setAutoFillBackground(False)
        self.documentationLink.setFrameShape(QFrame.NoFrame)
        self.documentationLink.setFrameShadow(QFrame.Plain)
        self.documentationLink.setLineWidth(0)
        self.documentationLink.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.documentationLink.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.documentationLink.setLineWrapMode(QTextEdit.NoWrap)
        self.documentationLink.setOpenExternalLinks(True)

        self.gridLayout.addWidget(self.documentationLink, 2, 1, 1, 1)

        self.versionDate_2 = QLabel(self.widget)
        self.versionDate_2.setObjectName(u"versionDate_2")
        sizePolicy.setHeightForWidth(self.versionDate_2.sizePolicy().hasHeightForWidth())
        self.versionDate_2.setSizePolicy(sizePolicy)
        self.versionDate_2.setMinimumSize(QSize(130, 26))
        self.versionDate_2.setMaximumSize(QSize(16777215, 26))
        self.versionDate_2.setFont(font)
        self.versionDate_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.versionDate_2, 3, 0, 1, 1)

        self.versionDateEdit_2 = QLabel(self.widget)
        self.versionDateEdit_2.setObjectName(u"versionDateEdit_2")
        sizePolicy.setHeightForWidth(self.versionDateEdit_2.sizePolicy().hasHeightForWidth())
        self.versionDateEdit_2.setSizePolicy(sizePolicy)
        self.versionDateEdit_2.setMinimumSize(QSize(0, 26))
        self.versionDateEdit_2.setMaximumSize(QSize(16777215, 26))
        self.versionDateEdit_2.setFont(font1)
        self.versionDateEdit_2.setStyleSheet(u"margin-left:1px;")
        self.versionDateEdit_2.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByMouse)

        self.gridLayout.addWidget(self.versionDateEdit_2, 3, 1, 1, 1)

        self.versionDate_3 = QLabel(self.widget)
        self.versionDate_3.setObjectName(u"versionDate_3")
        sizePolicy.setHeightForWidth(self.versionDate_3.sizePolicy().hasHeightForWidth())
        self.versionDate_3.setSizePolicy(sizePolicy)
        self.versionDate_3.setMinimumSize(QSize(130, 26))
        self.versionDate_3.setMaximumSize(QSize(16777215, 26))
        self.versionDate_3.setFont(font)
        self.versionDate_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.versionDate_3, 4, 0, 1, 1)

        self.versionDateEdit_3 = QLabel(self.widget)
        self.versionDateEdit_3.setObjectName(u"versionDateEdit_3")
        sizePolicy.setHeightForWidth(self.versionDateEdit_3.sizePolicy().hasHeightForWidth())
        self.versionDateEdit_3.setSizePolicy(sizePolicy)
        self.versionDateEdit_3.setMinimumSize(QSize(0, 26))
        self.versionDateEdit_3.setMaximumSize(QSize(16777215, 26))
        self.versionDateEdit_3.setFont(font1)
        self.versionDateEdit_3.setStyleSheet(u"margin-left:1px;")
        self.versionDateEdit_3.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByMouse)

        self.gridLayout.addWidget(self.versionDateEdit_3, 4, 1, 1, 1)


        self.gridLayout_2.addWidget(self.widget, 3, 0, 1, 1, Qt.AlignHCenter)

        self.verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 4, 0, 1, 1)

        self.softwareIcon = QLabel(Dialog)
        self.softwareIcon.setObjectName(u"softwareIcon")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.softwareIcon.sizePolicy().hasHeightForWidth())
        self.softwareIcon.setSizePolicy(sizePolicy2)
        self.softwareIcon.setMaximumSize(QSize(420, 140))
        self.softwareIcon.setPixmap(QPixmap(u"../static/edf2bids_full_icon.svg"))
        self.softwareIcon.setScaledContents(True)
        self.softwareIcon.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.softwareIcon, 0, 0, 1, 1, Qt.AlignHCenter)

        self.verticalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)

        self.gridLayout_2.addItem(self.verticalSpacer_2, 1, 0, 1, 1)

        self.closeAboutWindowButton = QPushButton(Dialog)
        self.closeAboutWindowButton.setObjectName(u"closeAboutWindowButton")
        sizePolicy2.setHeightForWidth(self.closeAboutWindowButton.sizePolicy().hasHeightForWidth())
        self.closeAboutWindowButton.setSizePolicy(sizePolicy2)
        self.closeAboutWindowButton.setFont(font1)

        self.gridLayout_2.addWidget(self.closeAboutWindowButton, 5, 0, 1, 1, Qt.AlignRight)


        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.versionDate.setText(QCoreApplication.translate("Dialog", u"Version date:", None))
        self.versionDateEdit.setText(QCoreApplication.translate("Dialog", u"00.00.000", None))
        self.googleDrive.setText(QCoreApplication.translate("Dialog", u"Google Drive:", None))
        self.googleDriveLink.setMarkdown(QCoreApplication.translate("Dialog", u"\n"
"[link to folder](https://drive.google.com/drive/folders/1op8Gv6sVWosIL7QQyXsUvNNjR5XXXj0j?usp=sharing)\n"
"\n"
"", None))
        self.googleDriveLink.setHtml(QCoreApplication.translate("Dialog", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Arial'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a href=\"https://drive.google.com/drive/folders/1op8Gv6sVWosIL7QQyXsUvNNjR5XXXj0j?usp=sharing\"><span style=\" font-size:12pt; text-decoration: underline; color:#0000ff;\">link to folder</span></a></p></body></html>", None))
        self.googleDriveLink.setPlaceholderText("")
        self.documentation.setText(QCoreApplication.translate("Dialog", u"Documentation: ", None))
        self.documentationLink.setMarkdown(QCoreApplication.translate("Dialog", u"[https://data2bids.greydongilmore.com](https://data2bids.greydongilmore.com)\n"
"\n"
"", None))
        self.documentationLink.setHtml(QCoreApplication.translate("Dialog", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Arial'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:8px; margin-bottom:8px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><a href=\"https://data2bids.greydongilmore.com\"><span style=\" font-size:12pt; text-decoration: underline; color:#0000ff;\">https://data2bids.greydongilmore.com</span></a></p></body></html>", None))
        self.documentationLink.setPlaceholderText("")
        self.versionDate_2.setText(QCoreApplication.translate("Dialog", u"Developer", None))
        self.versionDateEdit_2.setText(QCoreApplication.translate("Dialog", u"Greydon Gilmore", None))
        self.versionDate_3.setText(QCoreApplication.translate("Dialog", u"Email:", None))
        self.versionDateEdit_3.setText(QCoreApplication.translate("Dialog", u"greydon.gilmore@gmail.com", None))
        self.softwareIcon.setText("")
        self.closeAboutWindowButton.setText(QCoreApplication.translate("Dialog", u"Close", None))
    # retranslateUi

