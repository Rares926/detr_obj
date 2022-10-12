
from .position_encoding import build_position_encoding
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor
from darknet2torch import Darknet

class Yolo_BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # pe features vrea ultimul layer din model 
        return_layers = {'module_list': "13"}

        ####UNCOMMENT THIS LATER IT S JUST TO TEST WHERE DOES THE FORWARD ERROR CAME FROM 
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors,True)
        out: Dict[str, NestedTensor] = {}
        xs = {"features" : xs}      #workaround because we don t use intermediateLayerGetter 
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

class Yolo_backbone(Yolo_BackboneBase):
    """MobileNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # backbone = Darknet("")
        backbone = torch.load("/media/storagezimmer1/RaresPatrascu/projects/detR/detr_obj/workspace/saved_models/torch_model.pth")
        num_channels = 64
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def build_yolo_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Yolo_backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
