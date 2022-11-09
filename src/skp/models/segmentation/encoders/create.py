import re
import timm
import torch
import torch.nn as nn

from ...backbones import create_x3d
from ...tools import change_num_input_channels
from .swin_encoder import SwinTransformer


def get_attribute(model, name):
    """Hacked together function to retrieve the desired module from the model
    based on its string attribute name. But it works. 
    """
    name = name.split(".")
    for i, n in enumerate(name):
        if i == 0: 
            if isinstance(n, int):
                attr = model[n]
            else:
                attr = getattr(model, n)
        else:
            if isinstance(n, int):
                attr = attr[n]
            else:
                attr = getattr(attr, n)
    return attr


def check_if_int(s):
    try:
        _ = int(s)
        return True
    except ValueError:
        return False


def create_encoder(name, encoder_params, encoder_output_stride=32, in_channels=3):
    assert "pretrained" in encoder_params

    if name == "swin":
        assert encoder_output_stride == 32, "`swin` encoders only support output_stride=32"
        encoder = SwinTransformer(**encoder_params) 
    elif "x3d" in name:
        encoder = create_x3d(name, features_only=True, **encoder_params)
        assert encoder_output_stride in [16, 32]
        if encoder_output_stride == 16:
            encoder.model.blocks[-2].res_blocks[0].branch1_conv.stride = (1, 1, 1)
            encoder.model.blocks[-2].res_blocks[0].branch2.conv_b.stride = (1, 1, 1)
    else:
        encoder = timm.create_model(name, features_only=True, **encoder_params)
        encoder.out_channels = encoder.feature_info.channels()

        if encoder_output_stride != 32:
            # Default for pretty much every model is 32 
            # First, ensure that the provided stride is valid
            assert 32 % encoder_output_stride == 0
            scale_factor = 32 // encoder_output_stride 
            layers_to_modify = 1 if scale_factor == 2 else 2

            # First, get the layers with stride 2
            # For some models, there may be other conv layers with stride 2
            # that will need to be filtered out
            # EfficientNet is OK

            if re.search(r"resnest", name):
                if encoder_output_stride in [8, 16]:
                    encoder.layer4[0].downsample[0] = nn.Identity()
                    encoder.layer4[0].avd_last = nn.Identity()
                    if encoder_output_stride == 8:
                        encoder.layer3[0].downsample[0] = nn.Identity()
                        encoder.layer3[0].avd_last = nn.Identity()
                else:
                    raise Exception(f"{name} only supports output stride of 8, 16, or 32")

            elif re.search(r"resnet[0-9]+d", name):
                if encoder_output_stride in [8, 16]:
                    encoder.layer4[0].downsample[0] = nn.Identity()
                    encoder.layer4[0].conv1.stride = (1, 1)
                    encoder.layer4[0].conv2.stride = (1, 1)
                    if encoder_output_stride == 8:
                        encoder.layer3[0].downsample[0] = nn.Identity()
                        encoder.layer3[0].conv1.stride = (1, 1)
                        encoder.layer3[0].conv2.stride = (1, 1)
                else:
                    raise Exception(f"{name} only supports output stride of 8, 16, or 32")

            elif re.search(r"regnet[x|y]", name):
                downsample_convs = []
                for name, module in encoder.named_modules():
                    if hasattr(module, "stride"):
                        if module.stride == (2, 2):
                            downsample_convs += [name]

                downsample_convs = downsample_convs[::-1]
                for i in range(layers_to_modify * 2):
                    setattr(get_attribute(encoder, downsample_convs[i]), "stride", (1, 1))
               
            elif re.search(r"efficientnet|regnetz|rexnet", name):
                downsample_convs = []
                for name, module in encoder.named_modules():
                    if hasattr(module, "stride"):
                        if module.stride == (2, 2):
                            downsample_convs += [name]

                downsample_convs = downsample_convs[::-1]
                for i in range(layers_to_modify):
                    setattr(get_attribute(encoder, downsample_convs[i]), "stride", (1, 1))

            elif re.search(r"convnext", name):
                downsample_convs = []
                for name, module in encoder.named_modules():
                    if hasattr(module, "stride"):
                        if module.stride == (2, 2):
                            downsample_convs += [name]

                downsample_convs = downsample_convs[::-1]
                for i in range(layers_to_modify):
                    setattr(get_attribute(encoder, downsample_convs[i]), "stride", (1, 1))
                    # Need to also change the kernel size ...
                    # This involves creating a new layer with the appropriate kernel size
                    # Then modifying the weights to fit the new kernel size
                    # Then changing the layer in the model
                    in_channels = get_attribute(encoder, downsample_convs[i]).in_channels
                    out_channels = get_attribute(encoder, downsample_convs[i]).out_channels
                    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
                    w = get_attribute(encoder, downsample_convs[i]).weight
                    w = w.mean(-1, keepdim=True).mean(-2, keepdim=True)
                    conv_layer.weight = nn.Parameter(w)
                    split_name = downsample_convs[i].split(".")
                    if check_if_int(split_name[-1]):
                        # If the module name ends with a number that means it's within a sequential object
                        # and needs to be modified by accessing the module within a list.
                        #
                        # So you have to get the SEQUENTIAL object (by getting the attribute WITHOUT the number
                        # at the end) and then use that number as the list index and set the layer
                        # to that layer. Phew.
                        get_attribute(encoder, ".".join(split_name[:-1]))[int(split_name[-1])] = conv_layer
                    else:
                        # If the module name ends with a string that means it can be accessed by
                        # just grabbing the attribute
                        setattr(get_attribute(encoder, ".".join(split_name[:-1])), split_name[-1], conv_layer)


            else:
                raise Exception (f"{name} is not yet supported for output stride < 32")

    # Run a quick test to make sure the output stride is correct
    if "x3d" in name:
        x = torch.randn((2,3,64,64,64))
    else:
        x = torch.randn((2,3,128,128))
    final_out = encoder(x)[-1]
    actual_output_stride = x.size(-1) // final_out.size(-1)
    assert actual_output_stride == encoder_output_stride, f"Actual output stride [{actual_output_stride}] does not equal desired output stride [{encoder_output_stride}]"
    print(f"Confirmed encoder output stride {encoder_output_stride} !")
    encoder.output_stride = encoder_output_stride

    if in_channels != 3:
        encoder = change_num_input_channels(encoder, in_channels)

    return encoder
