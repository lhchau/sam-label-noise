from torch.optim import SGD
from .sam import SAM
from .samonly import SAMONLY
from .samwo import SAMWO
from .samen import SAMEN
from .gsamv2 import GSAMV2
from .powersamean import POWERSAMEAN
from .samenabs import SAMENABS
from .samenbc import SAMENBC
from .samenice import SAMENICE
from .samwoexplore import SAMWOEXPLORE
from .samlotteryticket import SAMLOTTERYTICKET
from .normsamean import NORMSAMEAN
from .normsam import NORMSAM

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
    elif opt_name == 'gsamv2':
        return GSAMV2(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samensq':
        return POWERSAMEAN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samenabs':
        return SAMENABS(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samenbc':
        return SAMENBC(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samenice':
        return SAMENICE(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samwoexplore':
        return SAMWOEXPLORE(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samlotteryticket':
        return SAMLOTTERYTICKET(net.parameters(), **opt_hyperpara)
    elif opt_name == 'normsamean':
        return NORMSAMEAN(net.parameters(), **opt_hyperpara)
    elif opt_name == 'normsam':
        return NORMSAM(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")