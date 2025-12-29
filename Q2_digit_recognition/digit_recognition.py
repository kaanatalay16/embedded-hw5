"""
EE 4065 - Embedded Digital Image Processing
Homework 5 - Question 2: Handwritten Digit Recognition from Digital Images

This script implements a handwritten digit recognition system using the MNIST dataset.
The trained model is converted to TensorFlow Lite format for deployment on STM32 microcontrollers.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
BATCH_SIZE = 128
EPOCHS = 20
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Split training into train and validation
    val_size = 5000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_data_augmentation():
    """Create data augmentation layer."""
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ])


def create_model(input_shape, num_classes):
    """Create CNN model for digit recognition."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(16, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def plot_training_history(history, save_path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Digit', fontsize=12)
    plt.ylabel('True Digit', fontsize=12)
    plt.title('Confusion Matrix - Handwritten Digit Recognition', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_sample_predictions(model, x_test, y_test, save_path, num_samples=25):
    """Plot sample predictions."""
    # Get predictions
    predictions = model.predict(x_test[:num_samples], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Create plot
    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].squeeze(), cmap='gray')
        color = 'green' if pred_labels[i] == y_test[i] else 'red'
        ax.set_title(f'Pred: {pred_labels[i]} (True: {y_test[i]})', 
                     color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def convert_to_tflite(model, x_train, save_path, quantize=True):
    """Convert Keras model to TensorFlow Lite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        def representative_dataset():
            for i in range(1000):
                yield [x_train[i:i+1].astype(np.float32)]
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {save_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    return tflite_model


def convert_to_c_array(tflite_model, output_path, array_name='mnist_model'):
    """Convert TFLite model to C header file."""
    with open(output_path, 'w') as f:
        f.write(f"// Auto-generated TFLite model header\n")
        f.write(f"// Model size: {len(tflite_model)} bytes\n\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(tflite_model)};\n")
        f.write(f"alignas(8) const unsigned char {array_name}[] = {{\n")
        
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        
        f.write("\n};\n\n")
        f.write(f"#endif // {array_name.upper()}_H\n")
    
    print(f"C header saved to {output_path}")


def evaluate_tflite_model(tflite_path, x_test, y_test):
    """Evaluate TFLite model accuracy."""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input/output quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    correct = 0
    total = len(x_test)
    
    for i in range(total):
        # Quantize input
        if input_details[0]['dtype'] == np.int8:
            input_data = (x_test[i:i+1] / input_scale + input_zero_point).astype(np.int8)
        else:
            input_data = x_test[i:i+1].astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        pred = np.argmax(output_data)
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / total
    print(f"TFLite Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    """Main training and conversion pipeline."""
    print("=" * 60)
    print("Handwritten Digit Recognition - Training Pipeline")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('../report/figures', exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    print("\n[2/7] Creating model...")
    model = create_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Create augmented training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss'
        ),
        keras.callbacks.ModelCheckpoint(
            'models/mnist_best.keras', save_best_only=True, monitor='val_accuracy'
        )
    ]
    
    # Train model
    print("\n[3/7] Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training history
    print("\n[4/7] Generating plots...")
    plot_training_history(history, 'figures/mnist_training_history.png')
    plot_training_history(history, '../report/figures/mnist_training_history.png')
    
    # Evaluate on test set
    print("\n[5/7] Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # Get predictions for confusion matrix
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    plot_confusion_matrix(y_test, y_pred, 'figures/mnist_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, '../report/figures/mnist_confusion_matrix.png')
    
    plot_sample_predictions(model, x_test, y_test, 'figures/mnist_predictions.png')
    plot_sample_predictions(model, x_test, y_test, '../report/figures/mnist_predictions.png')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=[str(i) for i in range(10)]))
    
    # Save Keras model
    print("\n[6/7] Saving models...")
    model.save('models/mnist_model.keras')
    
    # Convert to TFLite
    print("\n[7/7] Converting to TensorFlow Lite...")
    
    # Float32 version
    tflite_float = convert_to_tflite(
        model, x_train, 
        'models/mnist_model_float32.tflite',
        quantize=False
    )
    
    # Int8 quantized version
    tflite_int8 = convert_to_tflite(
        model, x_train,
        'models/mnist_model_int8.tflite',
        quantize=True
    )
    
    # Convert to C array
    convert_to_c_array(tflite_int8, 'models/mnist_model.h', 'mnist_model')
    
    # Evaluate TFLite model
    print("\nEvaluating quantized TFLite model...")
    evaluate_tflite_model('models/mnist_model_int8.tflite', x_test, y_test)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - models/mnist_model.keras (Keras model)")
    print(f"  - models/mnist_model_float32.tflite (Float32 TFLite)")
    print(f"  - models/mnist_model_int8.tflite (Quantized TFLite)")
    print(f"  - models/mnist_model.h (C header for STM32)")
    print(f"  - figures/mnist_training_history.png")
    print(f"  - figures/mnist_confusion_matrix.png")
    print(f"  - figures/mnist_predictions.png")


if __name__ == "__main__":
    main()

