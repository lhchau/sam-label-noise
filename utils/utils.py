'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def cosine_similarity(grad1, grad2):
    dot_product = torch.sum(grad1 * grad2)
    norm_grad1 = torch.norm(grad1)
    norm_grad2 = torch.norm(grad2)
    similarity = dot_product / (norm_grad1 * norm_grad2 + 1e-18)
    return similarity.item()

class HardBootstrappingLoss(nn.Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.

    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

def get_grads_and_masks_at_group(optimizer, gr='B', alpha=1):
    grads, masks = [], []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            param_state = optimizer.state[p]
            
            ratio = p.grad.div(param_state['first_grad'].add(1e-8))
            
            if gr == 'A':
                mask = ratio > alpha
            elif gr == 'B':
                mask = torch.logical_and(ratio < alpha, ratio > 0)
            else:
                mask = ratio <= 0
            masks.append(mask)
            grads.append(param_state['first_grad'].mul(mask))
    return grads, masks

def get_mask_A_B_same_or_diff_sign(grads_A, grads_B, sign='same'):
    masks = []
    for grad_A, grad_B in zip(grads_A, grads_B):
        if sign == 'same':
            mask = grad_A.mul(grad_B) > 0
        else: mask = grad_A.mul(grad_B) < 0
        masks.append(mask)
    return masks

def get_mask_A_less_magnitude_than_B_same_sign(grads_A, grads_B):
    masks = []
    for grad_A, grad_B in zip(grads_A, grads_B):
        masks.append(torch.logical_and(grad_A.abs() < grad_B.abs(), grad_A.mul(grad_B) > 0))
    return masks

def get_mask_A_less_magnitude_than_B_diff_sign(grads_A, grads_B):
    masks = []
    for grad_A, grad_B in zip(grads_A, grads_B):
        masks.append(torch.logical_and(grad_A.abs() < grad_B.abs(), grad_A.mul(grad_B) < 0))
    return masks

def count_overlap_two_mask(masksA, masksB):
    cnt = 0
    total_para = 0
    for maskA, maskB in zip(masksA, masksB):
        total_para += torch.sum(maskB)
        cnt += torch.sum(torch.logical_and(maskA, maskB))
    return cnt / total_para

def get_alpha(epoch, initial_alpha, final_alpha, total_epochs):
    if epoch < total_epochs:
        alpha = initial_alpha - (initial_alpha - final_alpha) * (epoch / total_epochs)
    else:
        alpha = final_alpha
    return alpha

def get_alpha_multi_step(epoch, curr_alpha, steps, gamma):
    if epoch in steps:
        alpha = curr_alpha * gamma
    else:
        alpha = curr_alpha
    return alpha

def get_gradients(optimizer):
    grads = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            grads.append(p.grad.clone())
    return grads

def get_norm(optimizer):
    logging_dict = {}
    if hasattr(optimizer, 'first_grad_norm'):
        logging_dict['first_grad_norm'] = optimizer.first_grad_norm
    if hasattr(optimizer, 'second_grad_norm'):
        logging_dict['second_grad_norm'] = optimizer.second_grad_norm
    if hasattr(optimizer, 'd_t_grad_norm'):
        logging_dict['d_t_grad_norm'] = optimizer.d_t_grad_norm
    return logging_dict
    
def get_checkpoint(optimizer, stored_info=[]):
    num_para_a, num_para_b, num_para_c, total_para = 0, 0, 0, 0
    if len(stored_info):
        ratios = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            param_state = optimizer.state[p]
            total_para += p.numel()
            
            ratio = p.grad.div(param_state['first_grad'].add(1e-8))
            if len(stored_info):
                ratios.append(ratio)
            num_para_a += torch.sum( ratio > 1 )
            num_para_b += torch.sum( torch.logical_and( ratio < 1, ratio > 0) )
            num_para_c += torch.sum( ratio <= 0)
    if len(stored_info):
        epoch = stored_info[0]
        if (epoch + 1) % 10 == 0:
            with open(f'./stored/ratios_epoch{epoch}.pkl', 'wb') as f:
                pickle.dump(ratios, f)
    return  {
        'num_para_a': (num_para_a / total_para) * 100, 
        'num_para_b': (num_para_b / total_para) * 100,
        'num_para_c': (num_para_c / total_para) * 100
    }

def get_logging_name(cfg):
    logging_name = ''
    
    logging_name += 'MOD'
    for key, value in cfg['model'].items():
        if isinstance(value, dict):
            for in_key, in_value in value.items():
                if isinstance(in_value, str):
                    _in_value = in_value[:5]
                else: _in_value = in_value
                logging_name += f'_{in_key[:2]}={_in_value}'
        else:
            logging_name += f'_{key[:2]}={value}'
        
    logging_name += '_OPT'
    for key, value in cfg['optimizer'].items():
        if isinstance(value, dict):
            for in_key, in_value in value.items():
                if isinstance(in_value, str):
                    _in_value = in_value[:5]
                else: _in_value = in_value
                logging_name += f'_{in_key[:2]}={_in_value}'
        else:
            logging_name += f'_{key[:2]}={value}'
        
    logging_name += '_DAT'
    for key, value in cfg['dataloader'].items():
        if isinstance(value, dict):
            for in_key, in_value in value.items():
                if isinstance(in_value, str):
                    _in_value = in_value[:5]
                else: _in_value = in_value
                logging_name += f'_{in_key[:2]}={_in_value}'
        else:
            logging_name += f'_{key[:2]}={value}'
        
    return logging_name

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    np.random.seed(seed)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 80  # default terminal width
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
