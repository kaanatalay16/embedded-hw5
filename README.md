# EE 4065 - Embedded Digital Image Processing - Homework 5

## Embedded Machine Learning Applications on STM32

This repository contains the implementation of two embedded machine learning applications for STM32 microcontrollers.

### Questions

1. **Q1: Keyword Spotting from Audio Signals** (Section 12.8)
   - Speech recognition using Google Speech Commands dataset
   - CNN-based model with MFCC feature extraction
   - Recognizes 10 keywords: yes, no, up, down, left, right, on, off, stop, go

2. **Q2: Handwritten Digit Recognition** (Section 12.9)
   - Image classification using MNIST dataset
   - Compact CNN model for digit recognition (0-9)
   - 98.7% accuracy with 72KB model size

### Results

| Application | Test Accuracy | Model Size (Int8) |
|-------------|--------------|-------------------|
| Keyword Spotting | 89.0% | 365 KB |
| Digit Recognition | 98.7% | 72 KB |

📄 **Full Report:** [report/main.pdf](report/main.pdf)

### Project Structure

```
embedded-hw5/
├── Q1_keyword_spotting/
│   ├── keyword_spotting.py      # Training script
│   ├── models/                  # Trained models
│   ├── figures/                 # Plots
│   └── stm32_deployment/        # STM32 code
├── Q2_digit_recognition/
│   ├── digit_recognition.py     # Training script
│   ├── models/                  # Trained models
│   ├── figures/                 # Plots
│   └── stm32_deployment/        # STM32 code
├── report/
│   ├── main.tex                 # LaTeX report source
│   ├── main.pdf                 # Report (PDF)
│   └── figures/                 # Report figures
└── requirements.txt             # Python dependencies
```

### Requirements

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Train Keyword Spotting model
cd Q1_keyword_spotting
python keyword_spotting.py

# Train Digit Recognition model
cd Q2_digit_recognition
python digit_recognition.py
```

### Reference

C. Ünsalan, B. Höke, and E. Atmaca, *Embedded Machine Learning with Microcontrollers: Applications on STM32 Boards*, Springer Nature, ISBN: 978-3031709111, 2025

