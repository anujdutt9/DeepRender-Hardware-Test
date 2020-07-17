# Hardware Engineer Recruitment Test
This repository contains two simple PyTorch models that have to be implemented for the iOS ecosystem using CoreML. Afterwards, the runtime performance of both models should be benchmarked (fps over the dataset) using an appropriate device or emulator.

The models are designed in such a way that they only include operations which are available in CoreML. Precisely, the following operations are needed:
1. Convolution Layer, See: https://apple.github.io/coremltools/coremlspecification/sections/NeuralNetwork.html#convolutionlayerparams
2. NN upsampling, See: https://apple.github.io/coremltools/coremlspecification/sections/NeuralNetwork.html#upsamplelayerparams
3. ActivationPReLU; See: https://apple.github.io/coremltools/coremlspecification/sections/NeuralNetwork.html#activationprelu

Each folder is standalone and contains one exercise. The python script uses PyTorch to transform a pre-calculated latent space, namely `y_hat`, into a predicted image, then compares the prediction to the `GT` and measures the `MSE` between both pictures. Each folder contains:
- A Python script containing the model-code
- A .pt file containing the model parameter
- A `y_hat` folder containing pre-calculated latent spaces.
- A `kodak` folder, containing ground truth images.

For more details, please check out the individual exercises / folders.
