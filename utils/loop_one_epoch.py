import torch
import os
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
    best_acc=0
    ):
    loss = 0
    total = 0
    correct = 0
    clean_total = 0
    clean_correct = 0
    noise_total = 0
    noise_correct = 0
    noise_acc, clean_acc = 0, 0
    
    if loop_type == 'train': 
        net.train()
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            elif len(batch) == 3:
                inputs, targets, noise_masks = batch
                inputs, targets, noise_masks = inputs.to(device), targets.to(device), noise_masks.to(device)
            
            if (batch_idx + 1) % len(dataloader) == 0:
                clean_inputs, noise_inputs = inputs[noise_masks == 0], inputs[noise_masks == 1]
                clean_targets, noise_targets = targets[noise_masks == 0], targets[noise_masks == 1]
                
                outputs = net(inputs)
                clean_outputs = outputs[torch.logical_not(noise_masks)]
                clean_targets = targets[torch.logical_not(noise_masks)]
                
                num_clean_examples = clean_inputs.shape[0]
                clean_loss = criterion(clean_outputs, clean_targets) * (num_clean_examples)
                clean_loss.backward()
                clean_grads = get_gradients(optimizer)
                optimizer.zero_grad()
                
                outputs = net(inputs)
                noise_outputs = outputs[noise_masks]
                noise_targets = targets[noise_masks]
                
                num_noise_examples = noise_inputs.shape[0]
                noise_loss = criterion(noise_outputs, noise_targets) * (num_noise_examples)
                noise_loss.backward()
                noise_grads = get_gradients(optimizer)
                optimizer.zero_grad()
                
                outputs = net(inputs)
                total_loss = criterion(outputs, targets) * len(outputs)
                total_loss.backward()
                total_grads = get_gradients(optimizer)
                optimizer.zero_grad()
                
            enable_running_stats(net)  # <- this is the important line
            outputs = net(inputs)
            first_loss = criterion(outputs, targets)
            first_loss.backward()        
            
            optimizer.first_step(zero_grad=True)
            
            disable_running_stats(net)  # <- this is the important line
            criterion(net(inputs), targets).backward()
            
            if (batch_idx + 1) % len(dataloader) == 0:
                logging_dict.update(get_checkpoint(optimizer))
                logging_dict.update(get_norm(optimizer))
                
                bad_masks = get_mask_A_less_magnitude_than_B_diff_sign(clean_grads, noise_grads)
                good_masks = get_mask_A_less_magnitude_than_B_diff_sign(noise_grads, clean_grads)
                _, masksA = get_grads_and_masks_at_group(optimizer, gr='A')
                _, masksB = get_grads_and_masks_at_group(optimizer, gr='B')
                _, masksC = get_grads_and_masks_at_group(optimizer, gr='C')
                
                clean_norm = calculate_norm(clean_grads, masksB)
                noise_norm = calculate_norm(noise_grads, masksB)
                total_norm = calculate_norm(total_grads, masksB)

                prop_A_over_bad = count_overlap_two_mask(masksA, bad_masks)
                prop_B_over_bad = count_overlap_two_mask(masksB, bad_masks)
                prop_C_over_bad = count_overlap_two_mask(masksC, bad_masks)
                
                prop_bad_over_A = count_overlap_two_mask(bad_masks, masksA)
                prop_bad_over_B = count_overlap_two_mask(bad_masks, masksB)
                prop_bad_over_C = count_overlap_two_mask(bad_masks, masksC)
                
                prop_good_over_A = count_overlap_two_mask(good_masks, masksA)
                prop_good_over_B = count_overlap_two_mask(good_masks, masksB)
                prop_good_over_C = count_overlap_two_mask(good_masks, masksC)
                
                prop_A_over_good = count_overlap_two_mask(masksA, good_masks)
                prop_B_over_good = count_overlap_two_mask(masksB, good_masks)
                prop_C_over_good = count_overlap_two_mask(masksC, good_masks)
                
                logging_dict.update({
                    'norm/clean_norm': clean_norm,
                    'norm/noise_norm': noise_norm,
                    'norm/total_norm': total_norm,
                    'prop/prop_A_over_bad': prop_A_over_bad,
                    'prop/prop_B_over_bad': prop_B_over_bad,
                    'prop/prop_C_over_bad': prop_C_over_bad,
                    'prop/prop_bad_over_A': prop_bad_over_A,
                    'prop/prop_bad_over_B': prop_bad_over_B,
                    'prop/prop_bad_over_C': prop_bad_over_C,
                    'prop/prop_good_over_A': prop_good_over_A,
                    'prop/prop_good_over_B': prop_good_over_B,
                    'prop/prop_good_over_C': prop_good_over_C,
                    'prop/prop_A_over_good': prop_A_over_good,
                    'prop/prop_B_over_good': prop_B_over_good,
                    'prop/prop_C_over_good': prop_C_over_good
                })
            
            optimizer.second_step(zero_grad=True)
            
            with torch.no_grad():
                loss += first_loss.item()
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
                    noise_acc = 100.*noise_correct/noise_total
                    
                    clean_total += (targets.size(0) - noise_masks.sum().item())
                    clean_correct += predicted.eq(targets).mul(torch.logical_not(noise_masks)).sum().item()
                    clean_acc = 100.*clean_correct/clean_total
                    
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Noise: %.3f%% (%d/%d) | Clean: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total, noise_acc, noise_correct, noise_total, clean_acc, clean_correct, clean_total))
        logging_dict[f'{loop_type.title()}/noise_acc'] = noise_acc
        logging_dict[f'{loop_type.title()}/clean_acc'] = clean_acc
        logging_dict[f'{loop_type.title()}/gap_clean_noise_acc'] = clean_acc - noise_acc
    else:
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += first_loss.item()
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
                
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'test': 
        return best_acc, acc