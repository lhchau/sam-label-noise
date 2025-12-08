import torch
import os
import numpy as np
from .utils import *
from .bypass_bn import *
import torch.nn.functional as F


def loop_one_epoch_jo(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type="train",
    logging_name=None,
    best_acc=0,
):
    """Run one epoch for training, testing, or evaluation."""

    # Tracking variables
    loss, total, correct = 0, 0, 0
    clean_total, clean_correct = 0, 0
    noise_total, noise_correct = 0, 0
    noise_acc, clean_acc = 0, 0

    if "train" in loop_type:
        net.train()
        results = np.zeros((len(dataloader.dataset), dataloader.dataset.num_classes), dtype=np.float32)

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, noise_masks, soft_targets, indexs = [x.to(device) for x in batch]

            opt_name = type(optimizer).__name__

            # --- SGD case ---
            if opt_name == "SGD":
                outputs = net(inputs)
                if loop_type != 'retrain':
                    probs, first_loss = criterion(outputs, soft_targets)
                    results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
                else:
                    first_loss = criterion(outputs, soft_targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # --- SAM-like optimizer case ---
            else:
                enable_running_stats(net)  # Important step
                outputs = net(inputs)

                optimizer.zero_grad()
                if loop_type != 'retrain':
                    probs, first_loss = criterion(outputs, soft_targets)
                    results[indexs.cpu().detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
                else:
                    first_loss = criterion(outputs, soft_targets)
                
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(net)  # Important step
                if loop_type != 'retrain':
                    _, second_loss = criterion(net(inputs), soft_targets)
                    second_loss.backward()
                else:
                    criterion(net(inputs), soft_targets).backward()

                optimizer.second_step(zero_grad=True)

            # --- Evaluation within training ---
            with torch.no_grad():
                loss += float(first_loss.item())
                loss_mean = loss / (batch_idx + 1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total

                if len(batch) == 2:  # No noise masks
                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")
                else:  # With noise masks
                    noise_total += noise_masks.sum().item()
                    noise_correct += predicted.eq(targets).mul(noise_masks).sum().item()
                    noise_acc = 100. * noise_correct / (noise_total + 1e-6)

                    clean_total += targets.size(0) - noise_masks.sum().item()
                    clean_correct += predicted.eq(targets).mul(~noise_masks).sum().item()
                    clean_acc = 100. * clean_correct / (clean_total + 1e-6)

                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total}) | "
                                     f"Noise: {noise_acc:.3f}% ({noise_correct}/{noise_total}) | "
                                     f"Clean: {clean_acc:.3f}% ({clean_correct}/{clean_total})")
        if loop_type != 'retrain':
            dataloader.dataset.label_update(results)

        # Save noise/clean accuracies
        logging_dict[f"Train/noise_acc"] = noise_acc
        logging_dict[f"Train/clean_acc"] = clean_acc
        logging_dict[f"Train/gap_clean_noise_acc"] = clean_acc - noise_acc

    # --- Testing phase ---
    elif loop_type == "test":
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

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

            # Save best checkpoint
            if acc > best_acc:
                print("Saving best checkpoint ...")
                state = {
                    "net": net.state_dict(),
                    "acc": acc,
                    "loss": loss,
                    "epoch": epoch,
                }
                save_path = os.path.join("checkpoint", logging_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "ckpt_best.pth"))
                best_acc = acc

            logging_dict[f"{loop_type.title()}/best_acc"] = best_acc

        logging_dict[f"{loop_type.title()}/gen_gap"] = logging_dict["Train/acc"] - acc

    # --- Resume from checkpoint ---
    else:
        print("==> Resuming from best checkpoint..")
        save_path = os.path.join("checkpoint", logging_name)
        checkpoint = torch.load(os.path.join(save_path, "ckpt_best.pth"))
        net.load_state_dict(checkpoint["net"])
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

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

    # Final logging
    if loop_type == 'retrain':
        logging_dict[f"Train/loss"] = loss_mean
        logging_dict[f"Train/acc"] = acc
    else:
        logging_dict[f"{loop_type.title()}/loss"] = loss_mean
        logging_dict[f"{loop_type.title()}/acc"] = acc

    if loop_type == "test":
        return best_acc, acc



def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type="train",
    logging_name=None,
    best_acc=0,
):
    """Run one epoch for training, testing, or evaluation."""

    # Tracking variables
    loss, total, correct = 0, 0, 0
    clean_total, clean_correct = 0, 0
    noise_total, noise_correct = 0, 0
    noise_acc, clean_acc = 0, 0
    noise_B_cosine_score, norm_grad_B, norm_noise_grad = [], [], []

    if loop_type == "train":
        net.train()

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, noise_masks = [x.to(device) for x in batch]
            noise_inputs, noise_targets = inputs[noise_masks == 1], targets[noise_masks == 1]

            opt_name = type(optimizer).__name__

            # --- SGD case ---
            if opt_name == "SGD":
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # --- SAM-like optimizer case ---
            else:
                enable_running_stats(net)  # Important step
                outputs = net(inputs)

                # Extra noise gradient logging every 8 batches
                if (batch_idx + 1) % 8 == 0:
                    noise_outputs = outputs[noise_masks]
                    batch_size = outputs.shape[0]
                    num_noise = noise_inputs.shape[0]
                    noise_loss = criterion(noise_outputs, noise_targets) * (num_noise / batch_size)
                    noise_loss.backward(retain_graph=True)
                    noise_grads = get_gradients(optimizer)

                optimizer.zero_grad()
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(net)  # Important step
                criterion(net(inputs), targets).backward()

                # Compare gradients between noise and B group
                if (batch_idx + 1) % 8 == 0:
                    B_grads, _ = get_grads_and_masks_at_group(optimizer)
                    for grad1, grad2 in zip(B_grads, noise_grads):
                        dot = torch.sum(grad1 * grad2)
                        norm1, norm2 = torch.norm(grad1), torch.norm(grad2)
                        cosine_sim = dot / (norm1 * norm2 + 1e-18)

                        norm_grad_B.append(norm1.item())
                        norm_noise_grad.append(norm2.item())
                        noise_B_cosine_score.append(cosine_sim.item())

                # Log gradient statistics at the end of each epoch
                if (batch_idx + 1) % len(dataloader) == 0:
                    logging_dict.update(get_checkpoint(optimizer))
                    logging_dict.update({
                        "prop/noise_B_cosine_score": np.mean(noise_B_cosine_score),
                        "prop/norm_grad_B": np.mean(norm_grad_B),
                        "prop/norm_noise_grad": np.mean(norm_noise_grad),
                    })

                optimizer.second_step(zero_grad=True)

            # --- Evaluation within training ---
            with torch.no_grad():
                loss += float(first_loss.item())
                loss_mean = loss / (batch_idx + 1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total

                if len(batch) == 2:  # No noise masks
                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")
                else:  # With noise masks
                    noise_total += noise_masks.sum().item()
                    noise_correct += predicted.eq(targets).mul(noise_masks).sum().item()
                    noise_acc = 100. * noise_correct / (noise_total + 1e-6)

                    clean_total += targets.size(0) - noise_masks.sum().item()
                    clean_correct += predicted.eq(targets).mul(~noise_masks).sum().item()
                    clean_acc = 100. * clean_correct / (clean_total + 1e-6)

                    if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                        progress_bar(batch_idx, len(dataloader),
                                     f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total}) | "
                                     f"Noise: {noise_acc:.3f}% ({noise_correct}/{noise_total}) | "
                                     f"Clean: {clean_acc:.3f}% ({clean_correct}/{clean_total})")

        # Save noise/clean accuracies
        logging_dict[f"{loop_type.title()}/noise_acc"] = noise_acc
        logging_dict[f"{loop_type.title()}/clean_acc"] = clean_acc
        logging_dict[f"{loop_type.title()}/gap_clean_noise_acc"] = clean_acc - noise_acc

    # --- Testing phase ---
    elif loop_type == "test":
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

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

            # Save best checkpoint
            if acc > best_acc:
                print("Saving best checkpoint ...")
                state = {
                    "net": net.state_dict(),
                    "acc": acc,
                    "loss": loss,
                    "epoch": epoch,
                }
                save_path = os.path.join("checkpoint", logging_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "ckpt_best.pth"))
                best_acc = acc

            logging_dict[f"{loop_type.title()}/best_acc"] = best_acc

        logging_dict[f"{loop_type.title()}/gen_gap"] = logging_dict["Train/acc"] - acc

    # --- Resume from checkpoint ---
    else:
        print("==> Resuming from best checkpoint..")
        save_path = os.path.join("checkpoint", logging_name)
        checkpoint = torch.load(os.path.join(save_path, "ckpt_best.pth"))
        net.load_state_dict(checkpoint["net"])
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

                loss_mean = loss / (batch_idx + 1)
                acc = 100. * correct / total

                if batch_idx % (len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

    # Final logging
    if loop_type == 'retrain':
        logging_dict[f"Train/loss"] = loss_mean
        logging_dict[f"Train/acc"] = acc
    else:
        logging_dict[f"{loop_type.title()}/loss"] = loss_mean
        logging_dict[f"{loop_type.title()}/acc"] = acc

    if loop_type == "test":
        return best_acc, acc
