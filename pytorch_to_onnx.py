import torch
from PIL import Image
import numpy as np 
import argparse
from models import build_model
from pathlib import Path
import io
import os
from datasets.custom import make_coco_transforms
import cv2
import onnxruntime
from util.misc import nested_tensor_from_tensor_list
from torch import nn, Tensor
import traceback

def get_args_parser():

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--num_classes', default=1, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--show_architecture', action='store_true',
                        help="If true, print model architecture")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="If true, only the backbone will be trained")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=60, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--freeze_transformer', action='store_true',
                        help="If true, only the transformer will be trained")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # * Onnx parameters
    parser.add_argument('--onnx_path', default='',
                        help='Path where to save model in the onnx format')

    return parser

def export_to_onnx(model, export_path, device):

    # generate model input
    dummy_image = torch.ones(1, 3, 640, 640)
    dummy_image = dummy_image.to(device)

    full_model_path = os.path.join(export_path, "detr_flat_bars.onnx")

    input_names = ["inputs"]
    output_names = ["pred_logits", "pred_boxes"]
    model.eval()

    # do_constant_folding will replace some of the ops that have all constant inputs with pre-computed constant nodes(optimization)
    # don t know why opset is 12 in docs 13 is the base so i guess some operation was changed in 13 and that is not ok with detr 

    torch.onnx.export(  model,
                        dummy_image,
                        full_model_path,
                        do_constant_folding=True,
                        opset_version=12,
                        input_names=input_names,
                        dynamic_axes=None,
                        output_names=output_names,
                        verbose=True)

def validate_exported_onnx_model(model, args):

    model.to("cpu")

    model.eval()

    model_path = os.path.join(args.onnx_path, "detr_flat_bars.onnx")

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()


    #create inputs
    test_inputs = [(torch.rand(1, 3, 640, 640),)]

    with torch.no_grad():
        detr_outs = model(*test_inputs[0])

    #- "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
    #                dictionnaries containing the two above keys for each decoder layer.

    if "aux_outputs" in detr_outs.keys():
        del detr_outs["aux_outputs"]

    detr_outs, _ = torch.jit._flatten(detr_outs)
    detr_outs = list(map(to_numpy, detr_outs))

    test_inputs, _ = torch.jit._flatten(test_inputs)
    test_inputs = list(map(to_numpy, test_inputs))

    #load onnx model 
    onnx_detr = onnxruntime.InferenceSession(model_path)

    ort_inputs = dict((onnx_detr.get_inputs()[i].name, inpt) for i, inpt in enumerate(test_inputs))

    onnx_detr_outs = onnx_detr.run(['pred_logits', 'pred_boxes'], ort_inputs)

    #rtol Relative tolerance.
    #atol Absolute tolerance (they must both be specified)
    #values for dtype float16 rtol 1e-3 atol 1e-5

    for i, element in enumerate(detr_outs):
        try:
            torch.testing.assert_allclose(element, onnx_detr_outs[i], rtol=1e-03, atol=1e-05)
        except AssertionError as error:
            traceback.print_exception(error)
        except Exception as er:
            traceback.print_exception(er)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('DETR export to onnx', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device(args.device)

    # build model 
    print("____Model build____")
    model, _, postprocessors = build_model(args)

    # load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        print("___Checkpoint path not given___")

    # move model on gpu
    model.to(device)

    # show model architecture 
    if args.show_architecture:
        print(model)

    # export to onnx
    export_to_onnx(model,args.onnx_path,device)

    #test onnx model 
    validate_exported_onnx_model(model, args)



