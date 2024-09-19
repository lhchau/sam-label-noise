from torch.optim import SGD
from .sam import SAM
from .samonly import SAMONLY
from .samwo import SAMWO
from .schedulersamean import SCHEDULERSAMEAN
from .fsam import FriendlySAM
from .fsamean import FriendlySAMEAN
from .vasso import VASSO
from .vassosamean import VASSOSAMEAN
from .gsam import GSAM
from .gsamean import GSAMEAN

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
    elif opt_name == 'schedulersamean':
        return SCHEDULERSAMEAN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'fsam':
        return FriendlySAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'fsamean':
        return FriendlySAMEAN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'vasso':
        return VASSO(net.parameters(), **opt_hyperpara)
    elif opt_name == 'vassosamean':
        return VASSOSAMEAN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'gsam':
        return GSAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'vassosamean':
        return GSAMEAN(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")