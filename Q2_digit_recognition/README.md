# Question 2: Handwritten Digit Recognition

This directory contains the implementation of a handwritten digit recognition system using the MNIST dataset.

## Overview

The system classifies handwritten digits (0-9) from 28×28 grayscale images.

## Files

- `digit_recognition.py` - Main training script
- `models/` - Trained models (Keras, TFLite, C header)
- `figures/` - Training plots and confusion matrix
- `stm32_deployment/` - STM32 deployment code

## Usage

### Training

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run training
python digit_recognition.py
```

### Output Files

After training, the following files are generated:

1. `models/mnist_model.keras` - Full Keras model
2. `models/mnist_model_float32.tflite` - Float32 TFLite model
3. `models/mnist_model_int8.tflite` - Quantized TFLite model
4. `models/mnist_model.h` - C header for STM32 deployment
5. `figures/mnist_training_history.png` - Training curves
6. `figures/mnist_confusion_matrix.png` - Confusion matrix
7. `figures/mnist_predictions.png` - Sample predictions

## Model Architecture

```
Layer                    Output Shape         Parameters
─────────────────────────────────────────────────────────
Conv2D (16, 3×3)        (26, 26, 16)         160
BatchNorm               (26, 26, 16)         64
MaxPool (2×2)           (13, 13, 16)         0
Conv2D (32, 3×3)        (11, 11, 32)         4,640
BatchNorm               (11, 11, 32)         128
MaxPool (2×2)           (5, 5, 32)           0
Conv2D (64, 3×3)        (3, 3, 64)           18,496
BatchNorm               (3, 3, 64)           256
Flatten                 (576)                0
Dense (64)              (64)                 36,928
Dropout (0.3)           (64)                 0
Dense (10)              (10)                 650
─────────────────────────────────────────────────────────
Total                                        ~61K
```

## Performance

- Test Accuracy: ~99%
- Model Size (Int8): ~62 KB
- Inference Time (STM32F4): ~12 ms

## STM32 Deployment

See `stm32_deployment/` directory for the embedded C code. The model is deployed using TensorFlow Lite for Microcontrollers.

### Requirements
- STM32F4 or higher with sufficient RAM (>30KB)
- Optional: Camera module or UART for image input

