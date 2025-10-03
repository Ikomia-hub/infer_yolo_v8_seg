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
import copy
import os

import torch

from ikomia import core, dataprocess, utils
import cv2

from ultralytics import YOLO
from ultralytics import download

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV8SegParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolov8m-seg"
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.7
        self.update = False
        self.model_weight_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "input_size": str(self.input_size),
            "conf_thres": str(self.conf_thres),
            "iou_thres": str(self.iou_thres),
            "model_weight_file": str(self.model_weight_file)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV8Seg(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV8SegParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.repo = 'ultralytics/assets'
        self.version = 'v0.0.0'
        self.device = torch.device("cpu")
        self.classes = None
        self.model = None
        self.half = False
        self.model_name = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def resize_to_stride(self, image, imgsz, stride=32):
        # Calculate the scaling factor based on the longer side
        scale_factor = imgsz / max(image.shape[:2])

        # Compute target dimensions (which might not be multiples of stride)
        target_width = int(scale_factor * image.shape[1])
        target_height = int(scale_factor * image.shape[0])

        # Adjust target dimensions for stride
        new_width = ((target_width + stride - 1) // stride) * stride
        new_height = ((target_height + stride - 1) // stride) * stride

        # Calculate width and height ratios (dw and dh)
        dw = image.shape[1] / new_width
        dh = image.shape[0] / new_height

        # Resize the image to the new dimensions
        resized_image = cv2.resize(image, (new_width, new_height))

        return resized_image, dw, dh

    def _load_model(self):
        param = self.get_param_object()
        self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        self.half = True if param.cuda and torch.cuda.is_available() else False

        if param.model_weight_file:
            self.model = YOLO(param.model_weight_file)
        else:
            # Set path
            model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
            model_weights = os.path.join(str(model_folder), f'{param.model_name}.pt')

            # Download model if not exist
            if not os.path.isfile(model_weights):
                url = f'https://github.com/{self.repo}/releases/download/{self.version}/{param.model_name}.pt'
                download(url=url, dir=model_folder, unzip=True)

            self.model = YOLO(model_weights)

        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Clean detection output
        self.get_output(1).clear_data()
        # Get parameters :
        param = self.get_param_object()
        # Get input :
        img_input = self.get_input(0)
        # Get image from input/output (numpy array):
        ini_src_image = img_input.get_image()

        # Resize image to input size and stride
        src_image, dw, dh = self.resize_to_stride(image=ini_src_image, imgsz=param.input_size)

       # Load model
        if param.update:
            self._load_model()

        # Run detection
        results = self.model.predict(
            src_image,
            save=False,
            imgsz=param.input_size,
            conf=param.conf_thres,
            iou=param.iou_thres,
            half=self.half,
            device=self.device
        )

        # Set classe names
        self.classes = list(results[0].names.values())
        self.set_names(self.classes)

        # Get output
        if results[0].masks is not None:
            boxes = results[0].boxes.xyxy
            confidences = results[0].boxes.conf
            class_idx = results[0].boxes.cls
            masks = results[0].masks.data
            masks = masks.detach().cpu().numpy()

            for i, (box, conf, cls, mask) in enumerate(zip(boxes, confidences, class_idx, masks)):
                box = box.detach().cpu().numpy()
                mask = cv2.resize(mask, ini_src_image.shape[:2][::-1])
                x1 = box[0] * dw
                x2 = box[2] * dw
                y1 = box[1] * dh
                y2 = box[3] * dh
                width = x2 - x1
                height = y2 - y1
                self.add_object(
                    i,
                    0,
                    int(cls),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(width),
                    float(height),
                    mask
                )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV8SegFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v8_seg"
        self.info.short_description = "Inference with YOLOv8 segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Instance Segmentation"
        self.info.version = "1.1.0"
        self.info.min_ikomia_version = "0.15.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "AGPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://docs.ultralytics.com/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_v8_seg"
        self.info.original_repository = "https://github.com/ultralytics/ultralytics"
        # Keywords used for search
        self.info.keywords = "YOLO, instance, segmentation, ultralytics"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INSTANCE_SEGMENTATION"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 8

    def create(self, param=None):
        # Create process object
        return InferYoloV8Seg(self.info.name, param)
