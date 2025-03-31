# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'settings_panel.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog,
    QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QSizePolicy, QSpacerItem, QTabWidget,
    QVBoxLayout, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(619, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QGridLayout(Dialog)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.tabWidget = QTabWidget(Dialog)
        self.tabWidget.setObjectName(u"tabWidget")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setDocumentMode(True)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_3 = QGridLayout(self.tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.widgetLab_2 = QWidget(self.tab)
        self.widgetLab_2.setObjectName(u"widgetLab_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(40)
        sizePolicy1.setHeightForWidth(self.widgetLab_2.sizePolicy().hasHeightForWidth())
        self.widgetLab_2.setSizePolicy(sizePolicy1)
        self.widgetLab_2.setMinimumSize(QSize(0, 0))
        self.widgetLab_2.setMaximumSize(QSize(16777215, 16777215))
        font1 = QFont()
        font1.setFamilies([u"Arial"])
        font1.setPointSize(11)
        self.widgetLab_2.setFont(font1)
        self.gridLayout = QGridLayout(self.widgetLab_2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.checkUpdatesLabel = QLabel(self.widgetLab_2)
        self.checkUpdatesLabel.setObjectName(u"checkUpdatesLabel")
        self.checkUpdatesLabel.setMinimumSize(QSize(120, 0))
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(11)
        font2.setBold(False)
        self.checkUpdatesLabel.setFont(font2)

        self.gridLayout.addWidget(self.checkUpdatesLabel, 0, 0, 1, 1)

        self.checkUpdates = QCheckBox(self.widgetLab_2)
        self.checkUpdates.setObjectName(u"checkUpdates")
        self.checkUpdates.setChecked(True)

        self.gridLayout.addWidget(self.checkUpdates, 0, 1, 1, 1)

        self.checkUpdatesLabel_2 = QLabel(self.widgetLab_2)
        self.checkUpdatesLabel_2.setObjectName(u"checkUpdatesLabel_2")
        self.checkUpdatesLabel_2.setMinimumSize(QSize(120, 0))
        self.checkUpdatesLabel_2.setFont(font2)

        self.gridLayout.addWidget(self.checkUpdatesLabel_2, 1, 0, 1, 1)

        self.recordingLabels = QLineEdit(self.widgetLab_2)
        self.recordingLabels.setObjectName(u"recordingLabels")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.recordingLabels.sizePolicy().hasHeightForWidth())
        self.recordingLabels.setSizePolicy(sizePolicy2)
        self.recordingLabels.setMinimumSize(QSize(300, 0))
        self.recordingLabels.setMaximumSize(QSize(300, 16777215))
        self.recordingLabels.setFont(font1)
        self.recordingLabels.setFrame(False)

        self.gridLayout.addWidget(self.recordingLabels, 1, 1, 1, 1)


        self.gridLayout_3.addWidget(self.widgetLab_2, 0, 0, 1, 1, Qt.AlignLeft)

        self.verticalSpacer_3 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)

        self.gridLayout_3.addItem(self.verticalSpacer_3, 1, 0, 1, 1)

        self.tabWidget.addTab(self.tab, "")
        self.tabJson = QWidget()
        self.tabJson.setObjectName(u"tabJson")
        self.verticalLayout = QVBoxLayout(self.tabJson)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widgetLab = QWidget(self.tabJson)
        self.widgetLab.setObjectName(u"widgetLab")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(40)
        sizePolicy3.setHeightForWidth(self.widgetLab.sizePolicy().hasHeightForWidth())
        self.widgetLab.setSizePolicy(sizePolicy3)
        self.widgetLab.setMinimumSize(QSize(0, 40))
        self.widgetLab.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_3 = QHBoxLayout(self.widgetLab)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.labelLab = QLabel(self.widgetLab)
        self.labelLab.setObjectName(u"labelLab")
        self.labelLab.setMinimumSize(QSize(120, 0))
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(10)
        font3.setBold(False)
        self.labelLab.setFont(font3)

        self.horizontalLayout_3.addWidget(self.labelLab)

        self.textboxLab = QLineEdit(self.widgetLab)
        self.textboxLab.setObjectName(u"textboxLab")
        self.textboxLab.setFont(font)
        self.textboxLab.setFrame(False)

        self.horizontalLayout_3.addWidget(self.textboxLab)


        self.verticalLayout.addWidget(self.widgetLab)

        self.widgetExperimenter = QWidget(self.tabJson)
        self.widgetExperimenter.setObjectName(u"widgetExperimenter")
        sizePolicy3.setHeightForWidth(self.widgetExperimenter.sizePolicy().hasHeightForWidth())
        self.widgetExperimenter.setSizePolicy(sizePolicy3)
        self.widgetExperimenter.setMinimumSize(QSize(0, 40))
        self.widgetExperimenter.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_2 = QHBoxLayout(self.widgetExperimenter)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.labelExperimenter = QLabel(self.widgetExperimenter)
        self.labelExperimenter.setObjectName(u"labelExperimenter")
        self.labelExperimenter.setMinimumSize(QSize(120, 0))
        self.labelExperimenter.setFont(font3)

        self.horizontalLayout_2.addWidget(self.labelExperimenter)

        self.textboxExperimenter = QLineEdit(self.widgetExperimenter)
        self.textboxExperimenter.setObjectName(u"textboxExperimenter")
        self.textboxExperimenter.setFont(font)
        self.textboxExperimenter.setFrame(False)

        self.horizontalLayout_2.addWidget(self.textboxExperimenter)


        self.verticalLayout.addWidget(self.widgetExperimenter)

        self.widgetDatasetName = QWidget(self.tabJson)
        self.widgetDatasetName.setObjectName(u"widgetDatasetName")
        sizePolicy3.setHeightForWidth(self.widgetDatasetName.sizePolicy().hasHeightForWidth())
        self.widgetDatasetName.setSizePolicy(sizePolicy3)
        self.widgetDatasetName.setMinimumSize(QSize(0, 40))
        self.widgetDatasetName.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout = QHBoxLayout(self.widgetDatasetName)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.labelDatasetName = QLabel(self.widgetDatasetName)
        self.labelDatasetName.setObjectName(u"labelDatasetName")
        self.labelDatasetName.setMinimumSize(QSize(120, 0))
        self.labelDatasetName.setFont(font3)

        self.horizontalLayout.addWidget(self.labelDatasetName)

        self.textboxDatasetName = QLineEdit(self.widgetDatasetName)
        self.textboxDatasetName.setObjectName(u"textboxDatasetName")
        self.textboxDatasetName.setFont(font)
        self.textboxDatasetName.setFrame(False)

        self.horizontalLayout.addWidget(self.textboxDatasetName)


        self.verticalLayout.addWidget(self.widgetDatasetName)

        self.widgetInstitutionAddress = QWidget(self.tabJson)
        self.widgetInstitutionAddress.setObjectName(u"widgetInstitutionAddress")
        sizePolicy3.setHeightForWidth(self.widgetInstitutionAddress.sizePolicy().hasHeightForWidth())
        self.widgetInstitutionAddress.setSizePolicy(sizePolicy3)
        self.widgetInstitutionAddress.setMinimumSize(QSize(0, 40))
        self.widgetInstitutionAddress.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_5 = QHBoxLayout(self.widgetInstitutionAddress)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.labelInstitutionAddress = QLabel(self.widgetInstitutionAddress)
        self.labelInstitutionAddress.setObjectName(u"labelInstitutionAddress")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(20)
        sizePolicy4.setHeightForWidth(self.labelInstitutionAddress.sizePolicy().hasHeightForWidth())
        self.labelInstitutionAddress.setSizePolicy(sizePolicy4)
        self.labelInstitutionAddress.setMinimumSize(QSize(120, 0))
        self.labelInstitutionAddress.setFont(font3)

        self.horizontalLayout_5.addWidget(self.labelInstitutionAddress)

        self.textboxInstitutionAddress = QLineEdit(self.widgetInstitutionAddress)
        self.textboxInstitutionAddress.setObjectName(u"textboxInstitutionAddress")
        self.textboxInstitutionAddress.setFont(font)
        self.textboxInstitutionAddress.setFrame(False)

        self.horizontalLayout_5.addWidget(self.textboxInstitutionAddress)


        self.verticalLayout.addWidget(self.widgetInstitutionAddress)

        self.widgetInstitutionName = QWidget(self.tabJson)
        self.widgetInstitutionName.setObjectName(u"widgetInstitutionName")
        sizePolicy3.setHeightForWidth(self.widgetInstitutionName.sizePolicy().hasHeightForWidth())
        self.widgetInstitutionName.setSizePolicy(sizePolicy3)
        self.widgetInstitutionName.setMinimumSize(QSize(0, 40))
        self.widgetInstitutionName.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_4 = QHBoxLayout(self.widgetInstitutionName)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.labelInstitutionName = QLabel(self.widgetInstitutionName)
        self.labelInstitutionName.setObjectName(u"labelInstitutionName")
        sizePolicy4.setHeightForWidth(self.labelInstitutionName.sizePolicy().hasHeightForWidth())
        self.labelInstitutionName.setSizePolicy(sizePolicy4)
        self.labelInstitutionName.setMinimumSize(QSize(120, 0))
        self.labelInstitutionName.setFont(font3)

        self.horizontalLayout_4.addWidget(self.labelInstitutionName)

        self.textboxInstitutionName = QLineEdit(self.widgetInstitutionName)
        self.textboxInstitutionName.setObjectName(u"textboxInstitutionName")
        self.textboxInstitutionName.setFont(font)
        self.textboxInstitutionName.setFrame(False)

        self.horizontalLayout_4.addWidget(self.textboxInstitutionName)


        self.verticalLayout.addWidget(self.widgetInstitutionName)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.tabWidget.addTab(self.tabJson, "")
        self.tabElectrodeInfo = QWidget()
        self.tabElectrodeInfo.setObjectName(u"tabElectrodeInfo")
        self.verticalLayout_2 = QVBoxLayout(self.tabElectrodeInfo)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.labeliEEG = QLabel(self.tabElectrodeInfo)
        self.labeliEEG.setObjectName(u"labeliEEG")
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(12)
        font4.setBold(True)
        self.labeliEEG.setFont(font4)

        self.verticalLayout_2.addWidget(self.labeliEEG)

        self.widgetIEEGManufacturer = QWidget(self.tabElectrodeInfo)
        self.widgetIEEGManufacturer.setObjectName(u"widgetIEEGManufacturer")
        sizePolicy3.setHeightForWidth(self.widgetIEEGManufacturer.sizePolicy().hasHeightForWidth())
        self.widgetIEEGManufacturer.setSizePolicy(sizePolicy3)
        self.widgetIEEGManufacturer.setMinimumSize(QSize(0, 40))
        self.widgetIEEGManufacturer.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_6 = QHBoxLayout(self.widgetIEEGManufacturer)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.labelIEEGManufacturer = QLabel(self.widgetIEEGManufacturer)
        self.labelIEEGManufacturer.setObjectName(u"labelIEEGManufacturer")
        self.labelIEEGManufacturer.setMinimumSize(QSize(120, 0))
        self.labelIEEGManufacturer.setFont(font3)
        self.labelIEEGManufacturer.setIndent(20)

        self.horizontalLayout_6.addWidget(self.labelIEEGManufacturer)

        self.textboxIEEGManufacturer = QLineEdit(self.widgetIEEGManufacturer)
        self.textboxIEEGManufacturer.setObjectName(u"textboxIEEGManufacturer")
        self.textboxIEEGManufacturer.setFont(font)
        self.textboxIEEGManufacturer.setFrame(False)

        self.horizontalLayout_6.addWidget(self.textboxIEEGManufacturer)


        self.verticalLayout_2.addWidget(self.widgetIEEGManufacturer)

        self.widgetIEEGType = QWidget(self.tabElectrodeInfo)
        self.widgetIEEGType.setObjectName(u"widgetIEEGType")
        sizePolicy3.setHeightForWidth(self.widgetIEEGType.sizePolicy().hasHeightForWidth())
        self.widgetIEEGType.setSizePolicy(sizePolicy3)
        self.widgetIEEGType.setMinimumSize(QSize(0, 40))
        self.widgetIEEGType.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_7 = QHBoxLayout(self.widgetIEEGType)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.labelIEEGType = QLabel(self.widgetIEEGType)
        self.labelIEEGType.setObjectName(u"labelIEEGType")
        self.labelIEEGType.setMinimumSize(QSize(120, 0))
        self.labelIEEGType.setFont(font3)
        self.labelIEEGType.setIndent(20)

        self.horizontalLayout_7.addWidget(self.labelIEEGType)

        self.textboxIEEGType = QLineEdit(self.widgetIEEGType)
        self.textboxIEEGType.setObjectName(u"textboxIEEGType")
        self.textboxIEEGType.setFont(font)
        self.textboxIEEGType.setFrame(False)

        self.horizontalLayout_7.addWidget(self.textboxIEEGType)


        self.verticalLayout_2.addWidget(self.widgetIEEGType)

        self.widgetIEEGMaterial = QWidget(self.tabElectrodeInfo)
        self.widgetIEEGMaterial.setObjectName(u"widgetIEEGMaterial")
        sizePolicy3.setHeightForWidth(self.widgetIEEGMaterial.sizePolicy().hasHeightForWidth())
        self.widgetIEEGMaterial.setSizePolicy(sizePolicy3)
        self.widgetIEEGMaterial.setMinimumSize(QSize(0, 40))
        self.widgetIEEGMaterial.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_8 = QHBoxLayout(self.widgetIEEGMaterial)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.labelIEEGMaterial = QLabel(self.widgetIEEGMaterial)
        self.labelIEEGMaterial.setObjectName(u"labelIEEGMaterial")
        self.labelIEEGMaterial.setMinimumSize(QSize(120, 0))
        self.labelIEEGMaterial.setFont(font3)
        self.labelIEEGMaterial.setIndent(20)

        self.horizontalLayout_8.addWidget(self.labelIEEGMaterial)

        self.textboxIEEGMaterial = QLineEdit(self.widgetIEEGMaterial)
        self.textboxIEEGMaterial.setObjectName(u"textboxIEEGMaterial")
        self.textboxIEEGMaterial.setFont(font)
        self.textboxIEEGMaterial.setFrame(False)

        self.horizontalLayout_8.addWidget(self.textboxIEEGMaterial)


        self.verticalLayout_2.addWidget(self.widgetIEEGMaterial)

        self.widgetIEEGDiameter = QWidget(self.tabElectrodeInfo)
        self.widgetIEEGDiameter.setObjectName(u"widgetIEEGDiameter")
        sizePolicy3.setHeightForWidth(self.widgetIEEGDiameter.sizePolicy().hasHeightForWidth())
        self.widgetIEEGDiameter.setSizePolicy(sizePolicy3)
        self.widgetIEEGDiameter.setMinimumSize(QSize(0, 40))
        self.widgetIEEGDiameter.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_9 = QHBoxLayout(self.widgetIEEGDiameter)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.labelIEEGDiameter = QLabel(self.widgetIEEGDiameter)
        self.labelIEEGDiameter.setObjectName(u"labelIEEGDiameter")
        self.labelIEEGDiameter.setMinimumSize(QSize(120, 0))
        self.labelIEEGDiameter.setFont(font3)
        self.labelIEEGDiameter.setIndent(20)

        self.horizontalLayout_9.addWidget(self.labelIEEGDiameter)

        self.textboxIEEGDiameter = QLineEdit(self.widgetIEEGDiameter)
        self.textboxIEEGDiameter.setObjectName(u"textboxIEEGDiameter")
        self.textboxIEEGDiameter.setFont(font)
        self.textboxIEEGDiameter.setFrame(False)

        self.horizontalLayout_9.addWidget(self.textboxIEEGDiameter)


        self.verticalLayout_2.addWidget(self.widgetIEEGDiameter)

        self.labelEEG = QLabel(self.tabElectrodeInfo)
        self.labelEEG.setObjectName(u"labelEEG")
        self.labelEEG.setFont(font4)

        self.verticalLayout_2.addWidget(self.labelEEG)

        self.widgetEEGManufacturer = QWidget(self.tabElectrodeInfo)
        self.widgetEEGManufacturer.setObjectName(u"widgetEEGManufacturer")
        sizePolicy3.setHeightForWidth(self.widgetEEGManufacturer.sizePolicy().hasHeightForWidth())
        self.widgetEEGManufacturer.setSizePolicy(sizePolicy3)
        self.widgetEEGManufacturer.setMinimumSize(QSize(0, 40))
        self.widgetEEGManufacturer.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_17 = QHBoxLayout(self.widgetEEGManufacturer)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.labelEEGManufacturer = QLabel(self.widgetEEGManufacturer)
        self.labelEEGManufacturer.setObjectName(u"labelEEGManufacturer")
        self.labelEEGManufacturer.setMinimumSize(QSize(120, 0))
        self.labelEEGManufacturer.setFont(font3)
        self.labelEEGManufacturer.setIndent(20)

        self.horizontalLayout_17.addWidget(self.labelEEGManufacturer)

        self.textboxEEGManufacturer = QLineEdit(self.widgetEEGManufacturer)
        self.textboxEEGManufacturer.setObjectName(u"textboxEEGManufacturer")
        self.textboxEEGManufacturer.setFont(font)
        self.textboxEEGManufacturer.setFrame(False)

        self.horizontalLayout_17.addWidget(self.textboxEEGManufacturer)


        self.verticalLayout_2.addWidget(self.widgetEEGManufacturer)

        self.widgetEEGType = QWidget(self.tabElectrodeInfo)
        self.widgetEEGType.setObjectName(u"widgetEEGType")
        sizePolicy3.setHeightForWidth(self.widgetEEGType.sizePolicy().hasHeightForWidth())
        self.widgetEEGType.setSizePolicy(sizePolicy3)
        self.widgetEEGType.setMinimumSize(QSize(0, 40))
        self.widgetEEGType.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_16 = QHBoxLayout(self.widgetEEGType)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.labelEEGType = QLabel(self.widgetEEGType)
        self.labelEEGType.setObjectName(u"labelEEGType")
        self.labelEEGType.setMinimumSize(QSize(120, 0))
        self.labelEEGType.setFont(font3)
        self.labelEEGType.setIndent(20)

        self.horizontalLayout_16.addWidget(self.labelEEGType)

        self.textboxEEGType = QLineEdit(self.widgetEEGType)
        self.textboxEEGType.setObjectName(u"textboxEEGType")
        self.textboxEEGType.setFont(font)
        self.textboxEEGType.setFrame(False)

        self.horizontalLayout_16.addWidget(self.textboxEEGType)


        self.verticalLayout_2.addWidget(self.widgetEEGType)

        self.widgetEEGMaterial = QWidget(self.tabElectrodeInfo)
        self.widgetEEGMaterial.setObjectName(u"widgetEEGMaterial")
        sizePolicy3.setHeightForWidth(self.widgetEEGMaterial.sizePolicy().hasHeightForWidth())
        self.widgetEEGMaterial.setSizePolicy(sizePolicy3)
        self.widgetEEGMaterial.setMinimumSize(QSize(0, 40))
        self.widgetEEGMaterial.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_14 = QHBoxLayout(self.widgetEEGMaterial)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.labelEEGMaterial = QLabel(self.widgetEEGMaterial)
        self.labelEEGMaterial.setObjectName(u"labelEEGMaterial")
        self.labelEEGMaterial.setMinimumSize(QSize(120, 0))
        self.labelEEGMaterial.setFont(font3)
        self.labelEEGMaterial.setIndent(20)

        self.horizontalLayout_14.addWidget(self.labelEEGMaterial)

        self.textboxEEGMaterial = QLineEdit(self.widgetEEGMaterial)
        self.textboxEEGMaterial.setObjectName(u"textboxEEGMaterial")
        self.textboxEEGMaterial.setFont(font)
        self.textboxEEGMaterial.setFrame(False)

        self.horizontalLayout_14.addWidget(self.textboxEEGMaterial)


        self.verticalLayout_2.addWidget(self.widgetEEGMaterial)

        self.widgetEEGDiameter = QWidget(self.tabElectrodeInfo)
        self.widgetEEGDiameter.setObjectName(u"widgetEEGDiameter")
        sizePolicy3.setHeightForWidth(self.widgetEEGDiameter.sizePolicy().hasHeightForWidth())
        self.widgetEEGDiameter.setSizePolicy(sizePolicy3)
        self.widgetEEGDiameter.setMinimumSize(QSize(0, 40))
        self.widgetEEGDiameter.setMaximumSize(QSize(16777215, 40))
        self.horizontalLayout_15 = QHBoxLayout(self.widgetEEGDiameter)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.labelEEGDiameter = QLabel(self.widgetEEGDiameter)
        self.labelEEGDiameter.setObjectName(u"labelEEGDiameter")
        self.labelEEGDiameter.setMinimumSize(QSize(120, 0))
        self.labelEEGDiameter.setFont(font3)
        self.labelEEGDiameter.setIndent(20)

        self.horizontalLayout_15.addWidget(self.labelEEGDiameter)

        self.textboxEEGDiameter = QLineEdit(self.widgetEEGDiameter)
        self.textboxEEGDiameter.setObjectName(u"textboxEEGDiameter")
        self.textboxEEGDiameter.setFont(font)
        self.textboxEEGDiameter.setFrame(False)

        self.horizontalLayout_15.addWidget(self.textboxEEGDiameter)


        self.verticalLayout_2.addWidget(self.widgetEEGDiameter)

        self.verticalSpacer_2 = QSpacerItem(20, 6, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_2)

        self.tabWidget.addTab(self.tabElectrodeInfo, "")

        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.buttonBoxJson = QDialogButtonBox(Dialog)
        self.buttonBoxJson.setObjectName(u"buttonBoxJson")
        self.buttonBoxJson.setFont(font)
        self.buttonBoxJson.setOrientation(Qt.Horizontal)
        self.buttonBoxJson.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout_2.addWidget(self.buttonBoxJson, 1, 0, 1, 1)


        self.retranslateUi(Dialog)
        self.buttonBoxJson.accepted.connect(Dialog.accept)
        self.buttonBoxJson.rejected.connect(Dialog.reject)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Settings", None))
        self.checkUpdatesLabel.setText(QCoreApplication.translate("Dialog", u"check for updates on startup", None))
        self.checkUpdates.setText("")
        self.checkUpdatesLabel_2.setText(QCoreApplication.translate("Dialog", u"Recording labels (comma list)", None))
        self.recordingLabels.setText(QCoreApplication.translate("Dialog", u"full,clip,stim,ccep", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Dialog", u"General settings", None))
        self.labelLab.setText(QCoreApplication.translate("Dialog", u"Lab", None))
        self.labelExperimenter.setText(QCoreApplication.translate("Dialog", u"Experimenter", None))
        self.labelDatasetName.setText(QCoreApplication.translate("Dialog", u"Dataset Name", None))
        self.labelInstitutionAddress.setText(QCoreApplication.translate("Dialog", u"Institution Address", None))
        self.labelInstitutionName.setText(QCoreApplication.translate("Dialog", u"Institution Name", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabJson), QCoreApplication.translate("Dialog", u"Json Metadata", None))
        self.labeliEEG.setText(QCoreApplication.translate("Dialog", u"iEEG Electrode Info:", None))
        self.labelIEEGManufacturer.setText(QCoreApplication.translate("Dialog", u"Manufacturer", None))
        self.labelIEEGType.setText(QCoreApplication.translate("Dialog", u"Type", None))
        self.labelIEEGMaterial.setText(QCoreApplication.translate("Dialog", u"Material", None))
        self.labelIEEGDiameter.setText(QCoreApplication.translate("Dialog", u"Diameter (mm)", None))
        self.labelEEG.setText(QCoreApplication.translate("Dialog", u"EEG Electrode Info:", None))
        self.labelEEGManufacturer.setText(QCoreApplication.translate("Dialog", u"Manufacturer", None))
        self.labelEEGType.setText(QCoreApplication.translate("Dialog", u"Type", None))
        self.labelEEGMaterial.setText(QCoreApplication.translate("Dialog", u"Material", None))
        self.labelEEGDiameter.setText(QCoreApplication.translate("Dialog", u"Diameter (mm)", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabElectrodeInfo), QCoreApplication.translate("Dialog", u"Electrode Info", None))
    # retranslateUi

