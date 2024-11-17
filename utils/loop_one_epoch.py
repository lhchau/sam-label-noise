import torch
import os
import numpy as np
from .utils import *
from .bypass_bn import *
import torch.nn.functional as F

def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type='train',
    logging_name=None,
    best_acc=0,
    log_fig4=False
    ):
    loss = 0
    total = 0
    correct = 0
    clean_total = 0
    clean_correct = 0
    noise_total = 0
    noise_correct = 0
    noise_acc, clean_acc = 0, 0
    if log_fig4:
        prop_A_over_bad_list, prop_B_over_bad_list, prop_C_over_bad_list = [], [], []
        prop_A_over_good_list, prop_B_over_good_list, prop_C_over_good_list = [], [], []
    
    if loop_type == 'train': 
        net.train()
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            elif len(batch) == 3:
                inputs, targets, noise_masks = batch
                inputs, targets, noise_masks = inputs.to(device), targets.to(device), noise_masks.to(device)
            if log_fig4:
                clean_inputs, noise_inputs = inputs[noise_masks == 0], inputs[noise_masks == 1]
                clean_targets, noise_targets = targets[noise_masks == 0], targets[noise_masks == 1]
                
            opt_name = type(optimizer).__name__
            if opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                if (batch_idx + 1) % 5 == 0 and log_fig4:
                    clean_outputs = outputs[torch.logical_not(noise_masks)]
                    clean_targets = targets[torch.logical_not(noise_masks)]
                    
                    num_clean_examples = clean_inputs.shape[0]
                    clean_loss = criterion(clean_outputs, clean_targets) * (num_clean_examples)
                    clean_loss.backward(retain_graph=True)
                    clean_grads = get_gradients(optimizer)
                    optimizer.zero_grad()
                    
                    noise_outputs = outputs[noise_masks]
                    noise_targets = targets[noise_masks]
                    
                    num_noise_examples = noise_inputs.shape[0]
                    noise_loss = criterion(noise_outputs, noise_targets) * (num_noise_examples)
                    noise_loss.backward(retain_graph=True)
                    noise_grads = get_gradients(optimizer)
                    
                optimizer.zero_grad()
                first_loss = criterion(outputs, targets)
                first_loss.backward()        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()

                if (batch_idx + 1) % 5 == 0 and log_fig4:
                    bad_masks = get_mask_A_less_magnitude_than_B_diff_sign(clean_grads, noise_grads)
                    good_masks = get_mask_A_less_magnitude_than_B_diff_sign(noise_grads, clean_grads)
                    _, masksA = get_grads_and_masks_at_group(optimizer, gr='A')
                    _, masksB = get_grads_and_masks_at_group(optimizer, gr='B')
                    _, masksC = get_grads_and_masks_at_group(optimizer, gr='C')
                    
                    prop_A_over_bad_list.append(count_overlap_two_mask(masksA, bad_masks).item())
                    prop_B_over_bad_list.append(count_overlap_two_mask(masksB, bad_masks).item())
                    prop_C_over_bad_list.append(count_overlap_two_mask(masksC, bad_masks).item())
                    
                    prop_A_over_good_list.append(count_overlap_two_mask(masksA, good_masks).item())
                    prop_B_over_good_list.append(count_overlap_two_mask(masksB, good_masks).item())
                    prop_C_over_good_list.append(count_overlap_two_mask(masksC, good_masks).item())

                if (batch_idx + 1) % len(dataloader) == 0:
                    logging_dict.update(get_checkpoint(optimizer))
                    logging_dict.update(get_norm(optimizer))

                    if log_fig4:
                        logging_dict.update({
                            'prop/prop_A_over_bad': np.mean(prop_A_over_bad_list),
                            'prop/prop_B_over_bad': np.mean(prop_B_over_bad_list),
                            'prop/prop_C_over_bad': np.mean(prop_C_over_bad_list),
                            'prop/prop_A_over_good': np.mean(prop_A_over_good_list),
                            'prop/prop_B_over_good': np.mean(prop_B_over_good_list),
                            'prop/prop_C_over_good': np.mean(prop_C_over_good_list)
                        })
                optimizer.second_step(zero_grad=True)
                
            with torch.no_grad():
                loss += float(first_loss.item())
                loss_mean = loss/(batch_idx+1)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total
                
                if len(batch) == 2:
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
                elif len(batch) == 3:
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = 100.*correct/total
                    
                    noise_total += noise_masks.sum().item()
                    noise_correct += predicted.eq(targets).mul(noise_masks).sum().item()
                    noise_acc = 100.*noise_correct/(noise_total + 1e-6)
                    
                    clean_total += (targets.size(0) - noise_masks.sum().item())
                    clean_correct += predicted.eq(targets).mul(torch.logical_not(noise_masks)).sum().item()
                    clean_acc = 100.*clean_correct/(clean_total + 1e-6)
                    
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Noise: %.3f%% (%d/%d) | Clean: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total, noise_acc, noise_correct, noise_total, clean_acc, clean_correct, clean_total))
        logging_dict[f'{loop_type.title()}/noise_acc'] = noise_acc
        logging_dict[f'{loop_type.title()}/clean_acc'] = clean_acc
        logging_dict[f'{loop_type.title()}/gap_clean_noise_acc'] = clean_acc - noise_acc
    elif loop_type == 'test':
        net.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
            if acc > best_acc:
                print('Saving best checkpoint ...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'loss': loss,
                    'epoch': epoch
                }
                save_path = os.path.join('checkpoint', logging_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
                best_acc = acc
            logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
        logging_dict[f'{loop_type.title()}/gen_gap'] = logging_dict['Train/acc'] - acc
    else:
        # Load checkpoint.
        print('==> Resuming from best checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        save_path = os.path.join('checkpoint', logging_name)
        checkpoint = torch.load(os.path.join(save_path, 'ckpt_best.pth'))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total

                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
                
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'test': 
        return best_acc, acc