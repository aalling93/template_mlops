import torch.nn as nn
import torch
import logging
from typing import Union, Dict, List
import gc


class ObjectDetector(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        number_classes: int = 2,
        weight: float = 1,
        stride: int = 32,
        dropout: float = 0.05,
        nr_scales: int = 3,
    ):
        """

        """

        super(ObjectDetector, self).__init__()

        self.strides = (stride // 4, stride // 2, stride)
        self.input_channels = input_channels
        self.verbose = 1
        self.number_classes = number_classes
        self.dropout = dropout

        self._background_threshold = 0.1


        self.training_mode = True
        self.export_mode = False
        self._post_process = False

        ##############################
        ##### Building the model #####
        ##############################


        ##############################
        #### post processing specks ####
        ##############################


    def forward(self, x, metadata=None) -> Union[torch.Tensor, list[torch.Tensor], List[torch.Tensor]]:

        x = self.pad_to_stride(x)
        features = self.backbone(x)
        neck = self.neck(features)
        outputs = self.head(neck, metadata)

        # x = self.pad_to_stride(x)
        #  features = self.backbone(x)
        #  #neck = self.neck(features)
        #  outputs = self.head(neck, metadata)

        # only entering if it is in eval model and post_processing is true
        if not self.training_mode and self._post_process:
            outputs = self.post_processing(outputs, x.shape)

        return outputs

    def train(self, training_mode=True):
        super().train(training_mode)
        self.training_mode = training_mode
        return self

    def eval(self):
        return self.train(False)

    def post_processing(self, outputs, img_size):
        if self.export_mode is False:

            return outputs

    def pad_to_stride(self, tensor):
        height, width = tensor.shape[-2:]
        pad_height = (self.strides[-1] - height % self.strides[-1]) * (height % self.strides[-1] != 0)
        pad_width = (self.strides[-1] - width % self.strides[-1]) * (width % self.strides[-1] != 0)
        # Padding format: (left, right, top, bottom)
        padded_tensor = nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

        return padded_tensor

    def export(
        self,
        file_path: str,
        metadata: Union[None, dict, Dict] = None,
        simplify_model: bool = False,
        opset_version: int = 11,
    ):
        self.export_mode = True

        x = torch.randn(1, self.input_channels, 256, 256).cpu()

        if file_path.endswith(".onnx"):
            logging.warning("ONNX gives nummerically different resutls! Better to use .pt")
            try:
                export_model_to_onnx(
                    self,
                    file_path,
                    x,
                    metadata=metadata,
                    simplify_model=simplify_model,
                    opset_version=opset_version,
                    verbose=self.verbose - 1,
                )
                self.export_type = "onnx"
            except Exception as e:
                logging.info(f"Error exporting model: {e}")

        elif file_path.endswith((".pth", ".torch", ".pytorch", ".pt")):
            try:
                traced_model = torch.jit.trace(self, x, strict=False)
                traced_model.save(file_path)
                self.export_type = "torch_jitter"
                del traced_model
                gc.collect()
            except Exception as e:
                logging.info(f"Error exporting model: {e}")

        return None


    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool):
        assert isinstance(value, bool), "The training_mode must be a boolean"
        self._training_mode = value
        return None

    @property
    def post_process(self):
        return self._post_process

    @post_process.setter
    def post_process(self, value: bool):
        self._post_process = value
        if value:
            self.eval()
        return None

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value: bool):
        self._export_mode = value
        if value:
            self.eval()
            self.cpu()
        else:
            self.train()
        return None

    @property
    def background_threshold(self):
        return self._background_threshold

    @background_threshold.setter
    def background_threshold(self, value: float):
        assert 0 <= value <= 1, "The background_threshold must be between 0 and 1"
        self._background_threshold = value
        return None

    @property
    def iou_threshold(self):
        return self._iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, value: float):
        assert 0 <= value <= 1, "The iou_threshold must be between 0 and 1"
        self._iou_threshold = value
        return None

    @property
    def skip_box_thr(self):
        return self._skip_box_thr

    @skip_box_thr.setter
    def skip_box_thr(self, value: float):
        assert 0 <= value <= 1, "The skip_box_thr must be between 0 and 1"
        self._skip_box_thr = value
        return None

    @property
    def conf_type(self):
        return self._conf_type

    @conf_type.setter
    def conf_type(self, value: str):
        assert value in ["max", "mean"], "The conf_type must be one of the following: 'max', 'mean'"
        self._conf_type = value
        return None
    
    
    def save_checkpoint(self, state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)


    def load_checkpoint(self, checkpoint, model, optimizer):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    
