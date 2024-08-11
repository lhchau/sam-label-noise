python train.py config/train_sam.yaml --optimizer.opt_name=samen --optimizer.group=B --optimizer.condition=0.5 --model.model_name=resnet18 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=sam-label-noise-cifar10

python train.py config/train_sam.yaml --optimizer.opt_name=samen --optimizer.group=B --optimizer.condition=2 --model.model_name=resnet18 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=sam-label-noise-cifar10

python train.py config/train_sam.yaml --optimizer.opt_name=samen --optimizer.group=B --optimizer.condition=0.5 --model.model_name=resnet18 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=sam-label-noise-cifar10

python train.py config/train_sam.yaml --optimizer.opt_name=samen --optimizer.group=B --optimizer.condition=2 --model.model_name=resnet18 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=sam-label-noise-cifar10

python train.py config/train_sam.yaml --optimizer.opt_name=samen --optimizer.group=B --optimizer.condition=0.5 --model.model_name=resnet18 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.75 --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=sam-label-noise-cifar10

python train.py config/train_sam.yaml --optimizer.opt_name=samen --optimizer.group=B --optimizer.condition=2 --model.model_name=resnet18 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.75 --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=sam-label-noise-cifar10
