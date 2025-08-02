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
# patience = cfg['trainer'].get('patience', 20)
scheduler = cfg['trainer'].get('scheduler', None)
use_val = cfg['dataloader'].get('use_val', False)

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
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
net = get_model(**cfg['model'], num_classes=num_classes)
net = net.to(device)
total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(net, opt_name, cfg['optimizer'])

if scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
elif scheduler == 'tiny_imagenet':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80])
elif scheduler == 'compare':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)])
# early_stopping = EarlyStopping(patience=patience)

################################
#### 3.b Training 
################################
if __name__ == "__main__":
    if resume:
        for epoch in range(1, start_epoch+1):
            scheduler.step()
    for epoch in range(start_epoch+1, EPOCHS+1):
        print('\nEpoch: %d' % epoch)
        loop_one_epoch(
            dataloader=train_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='train',
            logging_name=logging_name)
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
            best_acc=best_acc)
        if scheduler is not None:
            scheduler.step()
        
        if framework_name == 'wandb':
            wandb.log(logging_dict)
        elif framework_name == 'tensorboard':
            for metric_name, metric_value in logging_dict.items():
                writer.add_scalar(metric_name, metric_value, epoch)
                
        # if (epoch + 1) > 100:
        #     early_stopping(acc)
        #     if early_stopping.early_stop:
        #         break