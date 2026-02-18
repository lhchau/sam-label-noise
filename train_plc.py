import os
import wandb
import datetime
import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import *
from dataloader import *
from optimizer import *


################################
#### 0. SETUP CONFIGURATION
################################
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
cfg = exec_configurator()
initialize(cfg['trainer']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
best_acc, start_epoch, logging_dict = 0, 0, {}

EPOCHS = cfg['trainer']['epochs']
resume = cfg['trainer'].get('resume', None)
alpha_scheduler = cfg['trainer'].get('alpha_scheduler', None)
scheduler_name = cfg['trainer'].get('scheduler', None)
use_val = cfg['dataloader'].get('use_val', False)

# -------- PLC hyperparams (paper calls it Progressive Label Correction / PLC) --------
use_plc = cfg['trainer'].get('plc', True)
warmup_epochs = cfg['trainer'].get('warmup_epochs', 10)

# confidence threshold schedule (high -> lower)
tau_start = cfg['trainer'].get('tau_start', 0.90)      # very strict at first
tau_end   = cfg['trainer'].get('tau_end', 0.50)
tau_step  = cfg['trainer'].get('tau_step', 0.05)       # decrease per "round"
plc_rounds_per_tau = cfg['trainer'].get('rounds_per_tau', 1)  # how many epochs to run before decreasing tau

# optional: only flip up to some max fraction per epoch (stability)
max_flip_frac = cfg['trainer'].get('max_flip_frac', 1.0)  # 1.0 means no cap

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += f'_plc={use_plc}'
logging_name += ('_' + current_time)

framework_name = cfg['logging']['framework_name']
if framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)
elif framework_name == 'tensorboard':
    tb_log_dir = os.path.join('runs', cfg['logging']['project_name'], logging_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
pprint.pprint(cfg)

################################
#### 1. BUILD THE DATASET
################################
if use_val:
    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])
else:
    train_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])

train_dataset = train_dataloader.dataset  # must have mutable targets

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = get_model(**cfg['model'], num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(net, opt_name, cfg['optimizer'])

if scheduler_name == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
elif scheduler_name == 'tiny_imagenet':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80])
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)])


################################
#### PLC: Label correction pass
################################
@torch.no_grad()
def plc_correct_labels(
    dataloader,
    net,
    device,
    dataset,
    tau: float,
    num_classes: int,
    logging_dict: dict,
    max_flip_frac: float = 1.0,
):
    """
    Progressive Label Correction (PLC):
    Flip label y_i -> argmax p(.|x_i) only when model is very confident.

    Multi-class confidence criterion from the paper:
      gap = log p_max - log p_y (more robust than plain difference) :contentReference[oaicite:2]{index=2}
    Flip if:
      pred != y AND gap >= tau   (tau is a threshold on confidence gap)
    """
    net.eval()

    N = len(dataset.targets)
    flips = 0
    seen = 0

    # if dataloader yields (inputs, targets, noise_masks, indices) it's ideal.
    # If you only yield (inputs, targets, noise_masks), you MUST modify dataset to also return index.
    for batch in dataloader:
        if len(batch) == 4:
            inputs, targets, noise_masks, indices = batch
        elif len(batch) == 3:
            # no indices -> cannot update labels reliably
            raise ValueError("PLC needs dataset indices in each batch. Please return (inputs, targets, noise_masks, indices).")
        else:
            raise ValueError("Unexpected batch format for PLC.")

        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.to(device)

        logits = net(inputs)
        probs = torch.softmax(logits, dim=1)  # [B,C]
        pred = probs.argmax(dim=1)            # [B]

        # p_y: probability assigned to current label
        p_y = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # [B]
        p_max = probs.max(dim=1).values                         # [B]

        # robust gap (log difference) :contentReference[oaicite:3]{index=3}
        gap = torch.log(p_max + 1e-12) - torch.log(p_y + 1e-12)

        should_flip = (pred != targets) & (gap >= tau)

        # optional cap: do not flip too many labels in one pass
        if max_flip_frac < 1.0:
            max_flips = int(max_flip_frac * N)
        else:
            max_flips = N

        for i in range(inputs.size(0)):
            seen += 1
            if should_flip[i].item() and flips < max_flips:
                idx = int(indices[i].item())
                new_label = int(pred[i].item())
                dataset.targets[idx] = new_label
                flips += 1

    logging_dict["PLC/tau"] = float(tau)
    logging_dict["PLC/flips"] = int(flips)
    logging_dict["PLC/flip_rate"] = float(flips / max(1, N))
    return flips


################################
#### Training / Testing loop (PLC integrated)
################################
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
    loss, total, correct = 0, 0, 0
    clean_total, clean_correct = 0, 0
    noise_total, noise_correct = 0, 0
    noise_acc, clean_acc = 0.0, 0.0
    loss_mean, acc = 0.0, 0.0

    if loop_type == "train":
        net.train()
        for batch_idx, batch in enumerate(dataloader):
            # expect (inputs, targets, noise_masks, indices) or at least (inputs, targets, noise_masks)
            if len(batch) == 4:
                inputs, targets, noise_masks, _indices = batch
            else:
                inputs, targets, noise_masks = batch

            inputs, targets, noise_masks = inputs.to(device), targets.to(device), noise_masks.to(device)

            opt_name = type(optimizer).__name__

            if opt_name == "SGD":
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                enable_running_stats(net)
                outputs = net(inputs)

                optimizer.zero_grad()
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(net)
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                loss += float(first_loss.item())
                loss_mean = loss / (batch_idx + 1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.0 * correct / total

                noise_total += noise_masks.sum().item()
                noise_correct += predicted.eq(targets).mul(noise_masks.bool()).sum().item()
                noise_acc = 100.0 * noise_correct / (noise_total + 1e-6)

                clean_total += targets.size(0) - noise_masks.sum().item()
                clean_correct += predicted.eq(targets).mul((~noise_masks.bool())).sum().item()
                clean_acc = 100.0 * clean_correct / (clean_total + 1e-6)

                if batch_idx % max(1, (len(dataloader) // 10)) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total}) | "
                                 f"Noise: {noise_acc:.3f}% ({noise_correct}/{noise_total}) | "
                                 f"Clean: {clean_acc:.3f}% ({clean_correct}/{clean_total})")

        logging_dict["Train/loss"] = loss_mean
        logging_dict["Train/acc"] = acc
        logging_dict["Train/noise_acc"] = noise_acc
        logging_dict["Train/clean_acc"] = clean_acc
        logging_dict["Train/gap_clean_noise_acc"] = clean_acc - noise_acc

    elif loop_type == "test":
        net.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, targets = batch[:2]  # allow extra fields
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += float(first_loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss / (batch_idx + 1)
                acc = 100.0 * correct / total

                if batch_idx % max(1, (len(dataloader) // 10)) == 0 or (batch_idx + 1) == len(dataloader):
                    progress_bar(batch_idx, len(dataloader),
                                 f"Loss: {loss_mean:.3f} | Acc: {acc:.3f}% ({correct}/{total})")

            if acc > best_acc:
                print("Saving best checkpoint ...")
                state = {"net": net.state_dict(), "acc": acc, "loss": loss, "epoch": epoch}
                save_path = os.path.join("checkpoint", logging_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "ckpt_best.pth"))
                best_acc = acc

            logging_dict["Test/best_acc"] = best_acc
            logging_dict["Test/loss"] = loss_mean
            logging_dict["Test/acc"] = acc
            if "Train/acc" in logging_dict:
                logging_dict["Test/gen_gap"] = logging_dict["Train/acc"] - acc

        return best_acc, acc


################################
#### 3.b RUN
################################
if __name__ == "__main__":
    if resume:
        for epoch in range(1, start_epoch + 1):
            scheduler.step()

    # PLC threshold state
    tau = tau_start
    rounds_at_tau = 0

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        print('\nEpoch: %d' % epoch)

        # optional alpha scheduler you already had
        if alpha_scheduler:
            optimizer.set_alpha(get_alpha(epoch, initial_alpha=1, final_alpha=cfg['optimizer']['condition'], total_epochs=alpha_scheduler))

        # -------- TRAIN --------
        loop_one_epoch(
            dataloader=train_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='train',
            logging_name=logging_name
        )

        # -------- PLC CORRECTION (after warmup) --------
        if use_plc and epoch >= warmup_epochs:
            flips = plc_correct_labels(
                dataloader=train_dataloader,
                net=net,
                device=device,
                dataset=train_dataset,
                tau=tau,
                num_classes=num_classes,
                logging_dict=logging_dict,
                max_flip_frac=max_flip_frac,
            )

            # Decrease tau progressively (paper: start high, then lower) :contentReference[oaicite:4]{index=4}
            rounds_at_tau += 1
            if rounds_at_tau >= plc_rounds_per_tau:
                rounds_at_tau = 0
                tau = max(tau_end, tau - tau_step)

            # Optional: if nothing flips for a while, you can early-stop correction
            logging_dict["PLC/tau_next"] = float(tau)
            logging_dict["PLC/rounds_at_tau"] = int(rounds_at_tau)
            logging_dict["PLC/flips_epoch"] = int(flips)

        # -------- TEST --------
        best_acc, acc = loop_one_epoch(
            dataloader=test_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name,
            best_acc=best_acc
        )

        if scheduler is not None:
            scheduler.step()

        if framework_name == 'wandb':
            wandb.log(logging_dict)
        elif framework_name == 'tensorboard':
            for metric_name, metric_value in logging_dict.items():
                writer.add_scalar(metric_name, metric_value, epoch)
