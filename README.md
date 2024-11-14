## Setup

```
conda create -n sam python=3.11
conda activate sam
pip install -r requirements.txt
wandb login "WANDB_KEY"
```

## Run DenseNet121 

```
sh script/rebuttal_dn121_cifar10.sh
sh script/rebuttal_dn121_cifar100.sh
```