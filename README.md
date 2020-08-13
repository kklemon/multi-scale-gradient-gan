PyTorch Implementation of Multi-Scale Gradient GAN
==================================================

Work in progress. Training instructions, results and pre-trained models will be added soon.


TODO:
- [x] Visualization utils
- [x] Multi-GPU training
- [x] Different loss functions
- [ ] Different priors
- [ ] Deep extraction blocks
- [x] FP16 training
- [ ] R1 regularization of D
- [x] Refactor to PyTorch Lightning
- [x] Support with FFHQ dataset
- [ ] Compute FID scores
- [ ] Support classes
- [ ] Style GAN architecture
- [ ] Hyperparameter tuning towards following aspects:
    - [ ] Upsampling method (nearest, bilinear, convT)
    - [ ] Batch size
    - [ ] Loss function
    - [ ] Multi-GPU training (also towards time vs. results)
    - [ ] EMA generator
    - [ ] Weight initialization
    - [ ] Adam parameters
    - [ ] D regularization
    