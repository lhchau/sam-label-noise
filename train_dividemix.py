import wandb
import datetime
import pprint

import torch
import torch.nn as nn
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
# patience = cfg['trainer'].get('patience', 20)
scheduler = cfg['trainer'].get('scheduler', None)
use_val = cfg['dataloader'].get('use_val', False)

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += f'_k={alpha_scheduler}'
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

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net1 = get_model(**cfg['model'], num_classes=num_classes).to(device)
net2 = get_model(**cfg['model'], num_classes=num_classes).to(device)

total_params1 = sum(p.numel() for p in net1.parameters())
total_params2 = sum(p.numel() for p in net2.parameters())
print(f'==> Number of parameters (net1) in {cfg["model"]}: {total_params1}')
print(f'==> Number of parameters (net2) in {cfg["model"]}: {total_params2}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion_sup = nn.CrossEntropyLoss()
criterion_vec = nn.CrossEntropyLoss(reduction="none")

opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer1 = get_optimizer(net1, opt_name, cfg['optimizer'])
optimizer2 = get_optimizer(net2, opt_name, cfg['optimizer'])

if scheduler == 'cosine':
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)
elif scheduler == 'tiny_imagenet':
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[40, 80])
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[40, 80])
else:
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)])
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)])


################################
#### 3.b Training 
################################
if __name__ == "__main__":
    use_dividemix = cfg['trainer'].get('dividemix', True)
    warmup_epochs = cfg['trainer'].get('warmup_epochs', 10)
    p_threshold = cfg['trainer'].get('p_threshold', 0.5)
    lambda_u = cfg['trainer'].get('lambda_u', 25.0)
    T = cfg['trainer'].get('T', 0.5)
    alpha = cfg['trainer'].get('alpha', 4.0)

    # You need dataset length:
    n_train = len(train_dataloader.dataset)
    num_classes = num_classes  # already returned

    for epoch in range(start_epoch+1, EPOCHS+1):
        print('\nEpoch: %d' % epoch)

        if epoch <= warmup_epochs:
            # ---- warmup: standard training, both nets ----
            loop_one_epoch_warmup(train_dataloader, net1, optimizer1, device, criterion_sup, logging_dict, epoch, logging_name, tag="net1")
            loop_one_epoch_warmup(train_dataloader, net2, optimizer2, device, criterion_sup, logging_dict, epoch, logging_name, tag="net2")
        else:
            # ---- 1) estimate clean probabilities for each net ----
            losses1 = eval_loss_per_sample(net1, train_dataloader, device, criterion_vec, n_samples=n_train)
            losses2 = eval_loss_per_sample(net2, train_dataloader, device, criterion_vec, n_samples=n_train)

            p_clean1 = fit_gmm_two_components(losses1)
            p_clean2 = fit_gmm_two_components(losses2)

            # ---- 2) train net1 using net2's split, and net2 using net1's split ----
            train_dividemix_epoch(
                dataloader=train_dataloader,
                net=net1,
                net_other=net2,
                optimizer=optimizer1,
                device=device,
                num_classes=num_classes,
                p_clean_other=p_clean2,
                p_threshold=p_threshold,
                lambda_u=lambda_u,
                T=T,
                alpha=alpha,
                logging_dict=logging_dict,
                epoch=epoch,
                logging_name=logging_name,
                tag="net1",
            )

            train_dividemix_epoch(
                dataloader=train_dataloader,
                net=net2,
                net_other=net1,
                optimizer=optimizer2,
                device=device,
                num_classes=num_classes,
                p_clean_other=p_clean1,
                p_threshold=p_threshold,
                lambda_u=lambda_u,
                T=T,
                alpha=alpha,
                logging_dict=logging_dict,
                epoch=epoch,
                logging_name=logging_name,
                tag="net2",
            )

        # ---- test net1 (or ensemble if you want) ----
        best_acc, acc = loop_one_epoch(
            dataloader=test_dataloader,
            net=net1,
            criterion=criterion_sup,
            optimizer=optimizer1,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name,
            best_acc=best_acc
        )

        if scheduler1 is not None:
            scheduler1.step()
            scheduler2.step()

        if framework_name == 'wandb':
            wandb.log(logging_dict)
        else:
            for k, v in logging_dict.items():
                writer.add_scalar(k, v, epoch)
