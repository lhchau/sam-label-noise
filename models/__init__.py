from .resnet import *

def get_model(model_name, num_classes, widen_factor=1):
    if model_name == "resnet20":
        return resnet20(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet32":
        return resnet32(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet44":
        return resnet44(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet56":
        return resnet56(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet110":
        return resnet110(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet1202":
        return resnet1202(num_classes=num_classes, widen_factor=widen_factor)
    else:
        raise ValueError("Invalid model!!!")