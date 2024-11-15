# SAM
# python train.py config/train_sam.yaml --optimizer.opt_name=sam --model.model_name=resnet34 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=sam --model.model_name=resnet34 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# python train.py config/train_sam.yaml --optimizer.opt_name=sam --model.model_name=densenet121 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=sam --model.model_name=densenet121 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# python train.py config/train_sam.yaml --optimizer.opt_name=sam --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=sam --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# SGD-GrA
# python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=resnet34 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=resnet34 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=densenet121 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=densenet121 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

# SGD-GrB
# python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=resnet34 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=resnet34 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=densenet121 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=densenet121 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234


# python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234

python train.py config/train_sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=wideresnet40_2 --model.widen_factor=1 --dataloader.data_name=cifar100 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar100-label-noise-fig234