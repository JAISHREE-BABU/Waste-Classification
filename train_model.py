import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.model_training import WasteClassifier
from src.evaluation import ModelEvaluator
from config import *

def main():
    # Create directories
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    print("🚀 Starting Smart Waste Classifier Training...")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("Please download TrashNet dataset and place it in the data/raw/ folder")
        return
    
    # Step 1: Data Preparation
    print("\n📊 Step 1: Preparing Data...")
    preprocessor = DataPreprocessor()
    
    # Explore dataset
    class_counts = preprocessor.explore_dataset(DATASET_PATH)
    
    # Create data generators
    train_gen, val_gen = preprocessor.create_data_generators(DATASET_PATH)
    
    # Step 2: Model Creation
    print("\n🧠 Step 2: Creating Model...")
    classifier = WasteClassifier()
    
    # Choose model type (uncomment your choice)
    # model = classifier.create_custom_cnn()
    model, base_model = classifier.create_transfer_learning_model('mobilenetv2')
    
    # Compile model
    model = classifier.compile_model(model)
    model.summary()
    
    # Step 3: Training
    print("\n🎯 Step 3: Training Model...")
    history = classifier.train_model(model, train_gen, val_gen, epochs=EPOCHS)
    
    # Step 4: Save Model
    print("\n💾 Step 4: Saving Model...")
    model_path = classifier.save_model(model, "waste_classifier")
    
    # Step 5: Evaluation
    print("\n📈 Step 5: Evaluating Model...")
    evaluator = ModelEvaluator(model, val_gen)
    
    # Comprehensive evaluation
    predictions, pred_classes, true_classes = evaluator.evaluate_model()
    evaluator.plot_confusion_matrix(true_classes, pred_classes)
    evaluator.plot_training_history(history)
    evaluator.plot_sample_predictions()
    
    print(f"\n✅ Training completed! Model saved to: {model_path}")

if __name__ == "__main__":
    main()