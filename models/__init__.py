from .resnet import *
from .small_resnet import small_resnet20, small_resnet32, small_resnet44, small_resnet56, small_resnet110, small_resnet1202

def get_model(model_name, num_classes, widen_factor=1):
    if model_name == "resnet18":
        return resnet18(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet34":
        return resnet34(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet50":
        return resnet50(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet101":
        return resnet101(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "resnet152":
        return resnet152(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "small_resnet20":
        return small_resnet20(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "small_resnet32":
        return small_resnet32(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "small_resnet44":
        return small_resnet44(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "small_resnet56":
        return small_resnet56(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "small_resnet110":
        return small_resnet110(num_classes=num_classes, widen_factor=widen_factor)
    elif model_name == "small_resnet1202":
        return small_resnet1202(num_classes=num_classes, widen_factor=widen_factor)
    else:
        raise ValueError("Invalid model!!!")