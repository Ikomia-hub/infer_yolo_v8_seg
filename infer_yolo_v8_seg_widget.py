# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_yolo_v8_seg.infer_yolo_v8_seg_process import InferYoloV8SegParam
from PyQt5.QtWidgets import *
from torch.cuda import is_available


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferYoloV8SegWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYoloV8SegParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("yolov8n-seg")
        self.combo_model.addItem("yolov8s-seg")
        self.combo_model.addItem("yolov8m-seg")
        self.combo_model.addItem("yolov8l-seg")
        self.combo_model.addItem("yolov8x-seg")

        self.combo_model.setCurrentText(self.parameters.model_name)

        # Hyper-parameters
        custom_weight = bool(self.parameters.model_weight_file)
        self.check_cfg = QCheckBox("Custom model")
        self.check_cfg.setChecked(custom_weight)
        self.grid_layout.addWidget(
            self.check_cfg, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_cfg.stateChanged.connect(self.on_custom_weight_changed)

        self.label_hyp = QLabel("Model weight (.pt)")
        self.browse_weight_file = pyqtutils.BrowseFileWidget(
                                        path=self.parameters.model_weight_file,
                                        tooltip="Select file",
                                        mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_weight_file, row, 1)

        self.label_hyp.setVisible(custom_weight)
        self.browse_weight_file.setVisible(custom_weight)

        # Input size
        self.spin_input_size = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Input size",
                                            self.parameters.input_size
        )

        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        # Confidence IOU
        self.spin_iou_thres = pyqtutils.append_double_spin(
                                            self.grid_layout,
                                            "Confidence IoU",
                                            self.parameters.iou_thres,
                                            min=0.,
                                            max=1.,
                                            step=0.01,
                                            decimals=2
        )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_custom_weight_changed(self, int):
        self.label_hyp.setVisible(self.check_cfg.isChecked())
        self.browse_weight_file.setVisible(self.check_cfg.isChecked())

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.iou_thres = self.spin_iou_thres.value()
        if self.check_cfg.isChecked():
            self.parameters.model_weight_file = self.browse_weight_file.path
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferYoloV8SegWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolo_v8_seg"

    def create(self, param):
        # Create widget object
        return InferYoloV8SegWidget(param, None)
