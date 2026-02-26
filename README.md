# audio_classifier

Audio Signal Classification (SpeechCommands) – PyTorch

Overview
This project trains a small convolutional neural network (CNN) for audio command classification using the SpeechCommands dataset from torchaudio.

The script downloads the dataset automatically, extracts log-mel spectrogram features, applies basic data augmentation, handles class imbalance, and trains a CNN model end to end.

Features

Automatic dataset download (SpeechCommands via torchaudio)

16 kHz audio processing

Fixed 1-second audio clips (pad or trim)

Log-mel spectrogram feature extraction

CNN-based classifier

Data augmentation:

Time shifting

Additive noise

Simple spectrogram masking

MixUp (optional)

Class imbalance handling:

WeightedRandomSampler (default)

Optional class-weighted loss

Mixed precision training (AMP)

Optional torch.compile support

Basic inference speed check

Requirements

Python 3.9 or newer recommended

PyTorch

torchaudio

CUDA-enabled GPU optional (A100 works well but not required)

Make sure PyTorch and torchaudio are installed for your CUDA version if using GPU.

How to Run

Place sc_train.py in a folder and run:

python sc_train.py

What happens:

The dataset downloads into ./_sc_data

Training runs for the configured number of epochs

The best model is saved as best_sc.pt

Validation and test accuracy are printed

A small inference timing test runs at the end

Files Generated

./_sc_data/
Downloaded SpeechCommands dataset

best_sc.pt
Saved model checkpoint (weights + labels + config)

Configuration

Edit the RunCfg class inside sc_train.py to adjust training:

Important fields:

batch → batch size (increase for large GPUs)

workers → DataLoader workers

epochs → number of training epochs

lr → learning rate

use_sampler → enables imbalance sampling

use_weighted_loss → optional alternative imbalance method

mix_p / mix_a → MixUp settings

shift_p / noise_p / mask_p → augmentation probabilities

try_compile → enable torch.compile if supported

For large GPUs like A100:

Increase batch size

Increase num_workers

Keep AMP enabled for faster training

Model Details

Input:

1-second mono audio at 16 kHz

Feature extraction:

MelSpectrogram

Log amplitude conversion

Per-sample normalization

Architecture:

4 convolution blocks

BatchNorm + ReLU

Max pooling

Adaptive average pooling

Fully connected output layer

Inference

After training, the script:

Loads the best checkpoint

Runs evaluation on the test set

Performs a quick single-sample inference

Benchmarks average inference time (GPU only)

For real streaming inference (low-latency chunk-based prediction), the inference function can be extended to use sliding windows.

Notes

Using both sampler and weighted loss at the same time is usually unnecessary.

If torch.compile causes issues, disable it in RunCfg.

Dataset download happens only once and is cached locally.
