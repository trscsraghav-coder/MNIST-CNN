# MNIST Digit Classifier (CNN with PyTorch)

A convolutional neural network built with PyTorch that classifies handwritten digits (0-9) from the MNIST dataset.

## Result

**98.85% accuracy** on the 10,000-image test set.

## Requirements

- Python 3.10+
- torch
- torchvision
- matplotlib
- numpy
- pandas
- scikit-learn

Install with:
```
pip install torch torchvision matplotlib numpy pandas scikit-learn
```

For GPU acceleration (NVIDIA), install the CUDA build of PyTorch instead — see [pytorch.org](https://pytorch.org/get-started/locally/) and select your CUDA version.

## Dataset

MNIST is loaded directly through `torchvision.datasets.MNIST` (auto-downloads on first run):
- 60,000 training images
- 10,000 test images
- 28x28 grayscale, labeled 0-9

## Model Architecture

```
ConvolutionalNetwork(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

**Forward pass:**
1. Conv1 → ReLU → 2x2 max pool
2. Conv2 → ReLU → 2x2 max pool
3. Flatten to 400 features
4. Fully connected layers (400 → 120 → 84 → 10), ReLU between them
5. Log-softmax output (10 class probabilities)

## Training

- **Loss function:** Cross Entropy Loss
- **Optimizer:** Adam (learning rate 0.001)
- **Batch size:** 10
- **Epochs:** 5
- **Training time:** ~0.93 minutes (on GPU)

## Usage

Run the notebook (`mnist.ipynb`) top to bottom:
1. Loads and transforms the MNIST dataset into tensors
2. Wraps train/test data in `DataLoader`s
3. Defines and instantiates the CNN
4. Trains for 5 epochs, tracking loss and accuracy per epoch
5. Evaluates on the full test set
6. Includes a demo cell to visualize a single test image and check the model's prediction against it

## Notes

- `log_softmax` is used as the final activation combined with `CrossEntropyLoss`, matching PyTorch convention for multi-class classification.
- Training/test accuracy per epoch is plotted using matplotlib to visualize learning progress.
