## Sharpness Aware Minimization Exploration

### RQ 1: The performance of SAM versus SGD will be larger when we increase the width of ResNet models

1. Why it happened? 
- Previous research showed that SAM can drift along minima and find flatter minima. This effect happends in the later training stage when model achieve near zero loss.
- In general, WideResNet has a tendency to overfit more compared to ResNet. This is because Wideresnet uses wider layers with more channels, increasing the model's capacity. Higher capacity models have a greater ability to fit the training data, but this also means they are more prone to capturing noise and details specific to the training set, leading to overfitting.
- We conjecture that if the overfitting occur earlier, the drift along minima effect of SAM could be more effective.

2. Why is RQ interesting?
- It will be easier to achieve competitive performance with shallower and wider models.
- Rethinking the EfficientNet recipe.
- Extending to Transformers where do not restricted to local features and higher capacity.

3. How done now?
- Run SGD, SAM with 
    - ResNet18: widen_factor = 1, 1.25, 1.5, 2
    - ResNet34: widen_factor = 1, 1.25, 1.5, 2

Run command:
```{bash}
python train.py config/train_sam.yaml --optimizer.rho=0.05 --model.model_name=resnet18 --model.widen_factor=2 --logging.framework_name=None
```

4. What is missing?
- Which parts of SAM benefit wider ResNet?
- The effect of depth?

5. Actual results?

### RQ 2: The performance of SAM versus SGD will be smaller when we increase the depth of ResNet models

1. Why it happened? 
- Previous research showed that SAM can drift along minima and find flatter minima. This effect happends in the later training stage when model achieve near zero loss.
- In general, WideResNet has a tendency to overfit more compared to ResNet. This is because Wideresnet uses wider layers with more channels, increasing the model's capacity. Higher capacity models have a greater ability to fit the training data, but this also means they are more prone to capturing noise and details specific to the training set, leading to overfitting.
- We conjecture that if the overfitting occur later, the drift along minima effect of SAM could damage the convergence of model.

2. Why is RQ interesting?
- Proved that it is hard to train too deep model with SAM.

3. How done now?
- Run SGD, SAM with 
    - ResNet18
    - ResNet34
    - ResNet50
    - ResNet101
    - ResNet152

4. What is missing?
- Which parts of SAM exacberate deep ResNet?

5. Actual results?

### RQ 3: Why SAM benefits wider ResNet?

1. Why it happened? 

2. Why is RQ interesting?

3. How done now?

4. What is missing?

5. Actual results?