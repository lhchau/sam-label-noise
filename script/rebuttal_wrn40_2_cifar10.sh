# wideresnet40_2 k=25, alpha=0.25, 0.5, 0.75
python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.25 --trainer.alpha_scheduler=25 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.25 --trainer.alpha_scheduler=25 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.5 --trainer.alpha_scheduler=25 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.5 --trainer.alpha_scheduler=25 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.75 --trainer.alpha_scheduler=25 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.75 --trainer.alpha_scheduler=25 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

# wideresnet40_2 k=50, alpha=0.25, 0.5, 0.75
python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.25 --trainer.alpha_scheduler=50 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.25 --trainer.alpha_scheduler=50 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.5 --trainer.alpha_scheduler=50 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.5 --trainer.alpha_scheduler=50 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.75 --trainer.alpha_scheduler=50 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.75 --trainer.alpha_scheduler=50 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

# wideresnet40_2 k=50, alpha=0.25, 0.5, 0.75
python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.25 --trainer.alpha_scheduler=75 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.25 --trainer.alpha_scheduler=75 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.5 --trainer.alpha_scheduler=75 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.5 --trainer.alpha_scheduler=75 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.75 --trainer.alpha_scheduler=75 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal

python train.py config/train_sam.yaml --optimizer.opt_name=schedulersamean --optimizer.group=B --optimizer.condition=0.75 --trainer.alpha_scheduler=75 --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-rebuttal