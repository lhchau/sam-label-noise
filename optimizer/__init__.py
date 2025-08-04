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
from .samabs import SAMABS
from .samba import SAMBA
from .saner_last import SANERLAST
from .samba_last import SAMBALAST

def get_optimizer(
    net,
    opt_name='sam',
    last_linear_param_ids=None,
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
    elif opt_name == 'samabs':
        return SAMABS(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samba':
        return SAMBA(net.parameters(), **opt_hyperpara)
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
    elif opt_name == 'gsamean':
        return GSAMEAN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'sanerlast':
        if last_linear_param_ids is None:
            raise ValueError("last_linear_param_ids must be provided for SANERLAST optimizer")
        return SANERLAST(net.parameters(), last_linear_param_ids=last_linear_param_ids, **opt_hyperpara)
    elif opt_name == 'sambalast':
        if last_linear_param_ids is None:
            raise ValueError("last_linear_param_ids must be provided for SAMBALAST optimizer")
        return SAMBALAST(net.parameters(), last_linear_param_ids=last_linear_param_ids, **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")