import wandb
import datetime
import pprint

import torch
import torch.nn as nn

from models import *
from utils import *
from dataloader import *
from optimizer import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
cfg = exec_configurator()
initialize(cfg['trainer']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch, logging_dict = 0, 0, {}

# Total number of training epochs
EPOCHS = cfg['trainer']['epochs'] 

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += ('_' + current_time)

framework_name = cfg['logging']['framework_name']
if framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)
pprint.pprint(cfg)

resume = cfg['trainer'].get('resume', None)
################################
#### 1. BUILD THE DATASET
################################
train_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = get_model(**cfg['model'], num_classes=num_classes)
net = net.to(device)
if resume:
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    load_path = os.path.join('checkpoint', resume, 'ckpt_best.pth')
    checkpoint = torch.load(load_path)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['acc']

total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(net, opt_name, cfg['optimizer'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS * 0.5), int(EPOCHS * 0.75)])
early_stopping = EarlyStopping(patience=20)

################################
#### 3.b Training 
################################
if __name__ == "__main__":
    if resume:
        for epoch in range(0, start_epoch+1):
            scheduler.step()
    for epoch in range(start_epoch, start_epoch+EPOCHS):
        print('\nEpoch: %d' % epoch)
        if cfg['trainer']['alpha_scheduler']:
            optimizer.set_alpha(get_alpha(epoch, initial_alpha=1, final_alpha=cfg['optimizer']['condition'], total_epochs=EPOCHS))
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
        scheduler.step()
        
        if framework_name == 'wandb':
            wandb.log(logging_dict)
        if (epoch + 1) > 100:
            early_stopping(acc)
            if early_stopping.early_stop:
                break
            
    # if framework_name == 'wandb':
    #     wandb.log(logging_dict)
        
    #     mini_hessian_batch_size = 128
    #     cfg['dataloader']['batch_size'] = mini_hessian_batch_size
    #     train_dataloader, _, _ = get_dataloader(**cfg['dataloader'])
    #     figure = get_eigen_hessian_plot(
    #         name=logging_name, 
    #         net=net,
    #         criterion=criterion,
    #         dataloader=train_dataloader,
    #         hessian_batch_size=128*20,
    #         mini_hessian_batch_size=mini_hessian_batch_size
    #     )
    #     wandb.log({'train/top5_eigenvalue_density': wandb.Image(figure)})