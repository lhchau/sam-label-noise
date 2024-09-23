from .cifar100 import get_cifar100
from .cifar10 import get_cifar10
from .miniwebsion import get_miniwebvision
from .animal10n import get_animal10n

# Data
def get_dataloader(
    data_name='cifar10',
    batch_size=256,
    num_workers=4,
    noise=0.25,
    noise_type='symmetric',
    resize_image=224,
    data_augmentation="standard",
    data_size=1,
):
    print('==> Preparing data..')

    if data_name == "cifar100":
        return get_cifar100(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size)
    elif data_name == "cifar10":
        return get_cifar10(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size)
    elif data_name == "miniwebvision":
        return get_miniwebvision(batch_size, num_workers, resize_image)
    elif data_name == "animal10n":
        return get_animal10n(batch_size, num_workers)