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
criterion = nn.CrossEntropyLoss()

# IMPORTANT: per-sample criterion for co-teaching selection
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
    for epoch in range(start_epoch+1, EPOCHS+1):
        print('\nEpoch: %d' % epoch)
        if alpha_scheduler:
            optimizer1.set_alpha(get_alpha(epoch, initial_alpha=1, final_alpha=cfg['optimizer']['alpha'], total_epochs=alpha_scheduler))
            optimizer2.set_alpha(get_alpha(epoch, initial_alpha=1, final_alpha=cfg['optimizer']['alpha'], total_epochs=alpha_scheduler))

        use_coteaching = cfg['trainer'].get('coteaching', True)
        forget_rate = cfg['trainer'].get('forget_rate', 0.2)
        num_gradual = cfg['trainer'].get('num_gradual', 10)
        exponent = cfg['trainer'].get('exponent', 1.0)

        loop_one_epoch_co_teaching(
            dataloader=train_dataloader,
            net=(net1, net2) if use_coteaching else net1,
            criterion=criterion,
            optimizer=(optimizer1, optimizer2) if use_coteaching else optimizer1,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='train',
            logging_name=logging_name,
            coteaching=use_coteaching,
            criterion_vec=criterion_vec,
            forget_rate=forget_rate,
            num_gradual=num_gradual,
            exponent=exponent,
            total_epochs=EPOCHS,
        )

        best_acc, acc = loop_one_epoch_co_teaching(
            dataloader=test_dataloader,
            net=net1,                  # evaluate net1 (simple)
            criterion=criterion,
            optimizer=optimizer1,       # unused in test
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
        elif framework_name == 'tensorboard':
            for metric_name, metric_value in logging_dict.items():
                writer.add_scalar(metric_name, metric_value, epoch)
                
        # if (epoch + 1) > 100:
        #     early_stopping(acc)
        #     if early_stopping.early_stop:
        #         break