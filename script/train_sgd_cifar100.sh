python train.py config/train_sgd.yaml --model.model_name=resnet18_webvision --dataloader.data_name=miniwebvision --dataloader.batch_size=128 --trainer.seed=42 --trainer.epochs=120 --logging.framework_name=wandb --logging.project_name=miniwebvision-label-noise
python train.py config/train_sgd.yaml --model.model_name=resnet18_webvision --dataloader.data_name=miniwebvision --dataloader.batch_size=128 --trainer.seed=43 --trainer.epochs=120 --logging.framework_name=wandb --logging.project_name=miniwebvision-label-noise
python train.py config/train_sgd.yaml --model.model_name=resnet18_webvision --dataloader.data_name=miniwebvision --dataloader.batch_size=128 --trainer.seed=44 --trainer.epochs=120 --logging.framework_name=wandb --logging.project_name=miniwebvision-label-noise