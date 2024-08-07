from torch.optim import SGD
from .sam import SAM
from .samonly import SAMONLY
from .samwo import SAMWO
from .samen import SAMEN

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
    else:
        raise ValueError("Invalid optimizer!!!")