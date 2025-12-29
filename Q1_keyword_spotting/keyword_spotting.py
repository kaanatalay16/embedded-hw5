"""
EE 4065 - Embedded Digital Image Processing
Homework 5 - Question 1: Keyword Spotting from Audio Signals

This script implements a keyword spotting system using the Google Speech Commands dataset.
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
import pathlib
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
SAMPLE_RATE = 16000
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

# Target keywords (mini_speech_commands has 8 commands)
COMMANDS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


def download_dataset():
    """Download the Google Speech Commands dataset."""
    # Check multiple possible paths
    possible_paths = [
        pathlib.Path('data/mini_speech_commands'),
        pathlib.Path('data/mini_speech_commands_extracted/mini_speech_commands'),
    ]
    
    data_dir = None
    for p in possible_paths:
        if p.exists() and any(p.iterdir()):
            data_dir = p
            break
    
    if data_dir is None:
        print("Downloading Speech Commands dataset...")
        zip_path = keras.utils.get_file(
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            fname='mini_speech_commands.zip',
            extract=True,
            cache_dir='.',
            cache_subdir='data'
        )
        # Check paths again after download
        for p in possible_paths:
            if p.exists() and any(p.iterdir()):
                data_dir = p
                break
        
        if data_dir is None:
            # Try extracted path
            data_dir = pathlib.Path('data/mini_speech_commands_extracted/mini_speech_commands')
    
    # Clean up __MACOSX if exists
    for macosx_path in pathlib.Path('data').rglob('__MACOSX'):
        if macosx_path.exists():
            shutil.rmtree(macosx_path)
    
    print(f"Dataset directory: {data_dir}")
    return data_dir


def decode_audio(audio_binary):
    """Decode WAV file to audio tensor."""
    audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    """Extract label from file path."""
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    """Load audio file and extract label."""
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    """Convert waveform to spectrogram using STFT."""
    # Pad or truncate to exactly 1 second
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    
    # Compute STFT
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def preprocess_dataset(files, commands):
    """Preprocess dataset files."""
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    waveform_ds = files_ds.map(
        get_waveform_and_label, num_parallel_calls=AUTOTUNE
    )
    
    def make_spec_ds(waveform, label):
        spectrogram = get_spectrogram(waveform)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id
    
    spectrogram_ds = waveform_ds.map(
        make_spec_ds, num_parallel_calls=AUTOTUNE
    )
    return spectrogram_ds


def create_model(input_shape, num_classes):
    """Create CNN model for keyword spotting."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Preprocessing - normalize
        layers.Resizing(32, 32),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
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


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Keyword Spotting', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def convert_to_tflite(model, representative_dataset, save_path, quantize=True):
    """Convert Keras model to TensorFlow Lite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
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


def convert_to_c_array(tflite_model, output_path, array_name='kws_model'):
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


def main():
    """Main training and conversion pipeline."""
    print("=" * 60)
    print("Keyword Spotting - Training Pipeline")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('../report/figures', exist_ok=True)
    
    # Download and prepare dataset
    print("\n[1/7] Loading dataset...")
    data_dir = download_dataset()
    
    commands = np.array(COMMANDS)
    
    # Get all wav files, excluding __MACOSX and other unwanted folders
    all_files = []
    for cmd in COMMANDS:
        cmd_dir = data_dir / cmd
        if cmd_dir.exists():
            wav_files = list(cmd_dir.glob('*.wav'))
            all_files.extend([str(f) for f in wav_files])
    
    print(f"Found {len(all_files)} audio files")
    
    if len(all_files) == 0:
        print("ERROR: No audio files found!")
        print(f"Checking directory structure of {data_dir}:")
        if data_dir.exists():
            for item in data_dir.iterdir():
                print(f"  - {item.name}")
        return
    
    # Shuffle files
    np.random.shuffle(all_files)
    num_samples = len(all_files)
    
    # Split dataset
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    # Create datasets
    print("\n[2/7] Preprocessing datasets...")
    train_ds = preprocess_dataset(train_files, commands)
    val_ds = preprocess_dataset(val_files, commands)
    test_ds = preprocess_dataset(test_files, commands)
    
    # Get input shape
    for spec, _ in train_ds.take(1):
        input_shape = spec.shape
    print(f"Input shape: {input_shape}")
    
    # Batch and prefetch
    train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    
    # Create model
    print("\n[3/7] Creating model...")
    model = create_model(input_shape, len(commands))
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss'
        ),
        keras.callbacks.ModelCheckpoint(
            'models/kws_best.keras', save_best_only=True, monitor='val_accuracy'
        )
    ]
    
    # Train model
    print("\n[4/7] Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training history
    print("\n[5/7] Generating plots...")
    plot_training_history(history, 'figures/kws_training_history.png')
    plot_training_history(history, '../report/figures/kws_training_history.png')
    
    # Evaluate on test set
    print("\n[6/7] Evaluating model...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # Get predictions for confusion matrix
    y_true = []
    y_pred = []
    for specs, labels in test_ds:
        predictions = model.predict(specs, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    plot_confusion_matrix(y_true, y_pred, commands, 'figures/kws_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, commands, '../report/figures/kws_confusion_matrix.png')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=commands))
    
    # Convert to TFLite
    print("\n[7/7] Converting to TensorFlow Lite...")
    
    # Create representative dataset for quantization
    sample_data = []
    for specs, _ in train_ds.unbatch().take(500):
        sample_data.append(specs.numpy())
    sample_data = np.array(sample_data)
    
    def representative_dataset_gen():
        for i in range(min(500, len(sample_data))):
            yield [np.expand_dims(sample_data[i], axis=0).astype(np.float32)]
    
    # Save Keras model
    model.save('models/kws_model.keras')
    
    # Convert to TFLite (quantized)
    tflite_model = convert_to_tflite(
        model, 
        representative_dataset_gen, 
        'models/kws_model_int8.tflite',
        quantize=True
    )
    
    # Convert to C array
    convert_to_c_array(tflite_model, 'models/kws_model.h', 'kws_model')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - models/kws_model.keras (Keras model)")
    print(f"  - models/kws_model_int8.tflite (Quantized TFLite)")
    print(f"  - models/kws_model.h (C header for STM32)")
    print(f"  - figures/kws_training_history.png")
    print(f"  - figures/kws_confusion_matrix.png")


if __name__ == "__main__":
    main()
