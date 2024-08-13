from torch.optim import SGD
from .sam import SAM
from .samonly import SAMONLY
from .samwo import SAMWO
from .samen import SAMEN
from .samenu import SAMENU
from .gsamv2 import GSAMV2
from .customsam import CUSTOMSAM
from .samean import SAMEAN

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperpara={}):
    if opt_name == 'sam':
        return SAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'sgd':
        return SGD(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samonly':
        return SAMONLY(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samwo':
        return SAMWO(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samen':
        return SAMEN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samenu':
        return SAMENU(net.parameters(), **opt_hyperpara)
    elif opt_name == 'gsamv2':
        return GSAMV2(net.parameters(), **opt_hyperpara)
    elif opt_name == 'customsam':
        return CUSTOMSAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samean':
        return SAMEAN(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")