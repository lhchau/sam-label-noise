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

def calculate_norm(grads, mask):
    norm = torch.norm(
                torch.stack([
                    grad.mul(m).norm(p=2)
                    for grad, m in zip(grads, mask)
                ]),
                p=2
        )
    return norm

def get_group_B(optimizer, epoch):
    ratioB = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            param_state = optimizer.state[p]
            
            ratio = p.grad.div(param_state['first_grad'].add(1e-8))
            maskB = torch.logical_and(ratio < 1, ratio > 0)
            ratioB.append(ratio.mul(maskB))
    with open(f'./stored/ratioB_epoch{epoch}.pkl', 'wb') as f:
        pickle.dump(ratioB, f)

def check_total_grad(grads_A, grads_B, total_grads):
    for grad_A, grad_B, grads in zip(grads_A, grads_B, total_grads):
        mask = (grad_A.add(grad_B) - grads).abs() > 1e-6
        cnt = torch.sum(mask)
        if cnt > 0:
            print("Not equal")

def calculate_num_para_A_less_magnitude_than_B_same_sign(grads_A, grads_B, mask):
    cnt = 0
    total_para = sum([torch.sum(m) for m in mask])
    for grad_A, grad_B, m in zip(grads_A, grads_B, mask):
        magnitude_comparison = torch.logical_and(grad_A.abs() < grad_B.abs(), grad_A.mul(grad_B) > 0)
        masked_comparison = magnitude_comparison[m == 1]
        cnt += masked_comparison.sum().item()
    return (cnt / total_para) * 100

def calculate_num_para_A_less_magnitude_than_B(grads_A, grads_B, mask):
    cnt = 0
    total_para = sum([torch.sum(m) for m in mask])
    for grad_A, grad_B, m in zip(grads_A, grads_B, mask):
        magnitude_comparison = (grad_A.abs() < grad_B.abs())
        masked_comparison = magnitude_comparison[m == 1]
        cnt += masked_comparison.sum().item()
    return (cnt / total_para) * 100

def get_alpha(epoch, initial_alpha, final_alpha, total_epochs):
    if epoch < total_epochs:
        alpha = initial_alpha - (initial_alpha - final_alpha) * (epoch / total_epochs)
    else:
        alpha = final_alpha
    return alpha

def get_gradients(optimizer):
    grads = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            grads.append(p.grad.clone())
    return grads

def get_grads_and_masks_at_group(optimizer, gr='B'):
    grads, masks = [], []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            param_state = optimizer.state[p]
            
            ratio = p.grad.div(param_state['first_grad'].add(1e-8))
            
            if gr == 'A':
                mask = ratio > 1
            elif gr == 'B':
                mask = torch.logical_and(ratio < 1, ratio > 0)
            else:
                mask = ratio <= 0
            masks.append(mask)
            grads.append(p.grad.mul(mask))
    return grads, masks
    

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
        with open(f'./stored/ratios_epoch{epoch}.pkl', 'wb') as f:
            pickle.dump(ratios, f)
    return {
        'num_para_a': (num_para_a / total_para) * 100,
        'num_para_b': (num_para_b / total_para) * 100,
        'num_para_c': (num_para_c / total_para) * 100
    }
    

def get_logging_name(cfg):
    logging_name = ''
    
    logging_name += 'MODEL'
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
        
    logging_name += '_DATA'
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
