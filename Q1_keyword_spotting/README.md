# Question 1: Keyword Spotting from Audio Signals

This directory contains the implementation of a keyword spotting system using the Google Speech Commands dataset.

## Overview

The system recognizes 10 keywords: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"

## Files

- `keyword_spotting.py` - Main training script
- `models/` - Trained models (Keras, TFLite, C header)
- `figures/` - Training plots and confusion matrix
- `stm32_deployment/` - STM32 deployment code

## Usage

### Training

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run training
python keyword_spotting.py
```

### Output Files

After training, the following files are generated:

1. `models/kws_model.keras` - Full Keras model
2. `models/kws_model_int8.tflite` - Quantized TFLite model
3. `models/kws_model.h` - C header for STM32 deployment
4. `figures/kws_training_history.png` - Training curves
5. `figures/kws_confusion_matrix.png` - Confusion matrix

## Model Architecture

```
Layer                    Output Shape         Parameters
─────────────────────────────────────────────────────────
Conv2D (32, 3×3)        (47, 38, 32)         320
BatchNorm               (47, 38, 32)         128
MaxPool (2×2)           (23, 19, 32)         0
Conv2D (64, 3×3)        (21, 17, 64)         18,496
BatchNorm               (21, 17, 64)         256
MaxPool (2×2)           (10, 8, 64)          0
Conv2D (128, 3×3)       (8, 6, 128)          73,856
BatchNorm               (8, 6, 128)          512
MaxPool (2×2)           (4, 3, 128)          0
Flatten                 (1536)               0
Dense (128)             (128)                196,736
Dropout (0.3)           (128)                0
Dense (10)              (10)                 1,290
─────────────────────────────────────────────────────────
Total                                        ~291K
```

## STM32 Deployment

See `stm32_deployment/` directory for the embedded C code. The model is deployed using TensorFlow Lite for Microcontrollers.

### Requirements
- STM32F4 or higher with sufficient RAM (>100KB)
- I2S microphone (e.g., INMP441)
- CMSIS-DSP library for MFCC computation

