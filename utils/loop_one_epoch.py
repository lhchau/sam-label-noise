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
    ):
    loss = 0
    total = 0
    correct = 0
    clean_total = 0
    clean_correct = 0
    noise_total = 0
    noise_correct = 0
    noise_acc, clean_acc = 0, 0
    noise_B_cosine_score, norm_grad_B, norm_noise_grad = [], [], []
    if loop_type == 'train': 
        net.train()
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, noise_masks = batch
            inputs, targets, noise_masks = inputs.to(device), targets.to(device), noise_masks.to(device)
                
            noise_inputs, noise_targets = inputs[noise_masks == 1], targets[noise_masks == 1]
                
            opt_name = type (optimizer).__name__
            if opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                
                if (batch_idx + 1) % 8 == 0:
                    noise_outputs = outputs[noise_masks]
                    noise_targets = targets[noise_masks]
                    
                    num_noise_examples = noise_inputs.shape[0]
                    noise_loss = criterion(noise_outputs, noise_targets) * (num_noise_examples/128)
                    noise_loss.backward(retain_graph=True)
                    noise_grads = get_gradients(optimizer)
                
                optimizer.zero_grad()
                first_loss = criterion(outputs, targets)
                first_loss.backward()        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                
                if (batch_idx + 1) % 8 == 0:
                    B_grads, _ = get_grads_and_masks_at_group(optimizer)
                    for grad1, grad2 in zip(B_grads, noise_grads):
                        dot_product = torch.sum(grad1 * grad2)
                        norm_grad1 = torch.norm(grad1)
                        norm_grad2 = torch.norm(grad2)
                        similarity = dot_product / (norm_grad1 * norm_grad2 + 1e-18)

                        norm_grad_B.append(norm_grad1.item())
                        norm_noise_grad.append(norm_grad2.item())
                        noise_B_cosine_score.append(similarity.item())
                    if epoch == 1:
                        breakpoint()
                if (batch_idx + 1) % len(dataloader) == 0:
                    logging_dict.update(get_checkpoint(optimizer))

                    logging_dict.update({
                        'prop/noise_B_cosine_score': np.mean(noise_B_cosine_score),
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
                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
                else:
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = 100.*correct/total
                    
                    noise_total += noise_masks.sum().item()
                    noise_correct += predicted.eq(targets).mul(noise_masks).sum().item()
                    noise_acc = 100.*noise_correct/(noise_total + 1e-6)
                    
                    clean_total += (targets.size(0) - noise_masks.sum().item())
                    clean_correct += predicted.eq(targets).mul(torch.logical_not(noise_masks)).sum().item()
                    clean_acc = 100.*clean_correct/(clean_total + 1e-6)
                    
                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
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
                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
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
                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss_mean, acc, correct, total))
                
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'test': 
        return best_acc, acc