# Configuration file for the project
import os

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset-resized")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "trained_models")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "model_checkpoints")

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30  # Reduced for faster training
NUM_CLASSES = 6

# Class names
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Training parameters
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1