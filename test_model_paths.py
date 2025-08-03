import os
import tensorflow as tf

def test_model_paths():
    """Test different possible paths for the model file"""
    possible_paths = [
        'app/models/brain_tumor_classifier.h5',
        'models/brain_tumor_classifier.h5',
        '../models/brain_tumor_classifier.h5',
        './models/brain_tumor_classifier.h5'
    ]
    
    print("Testing model file paths...")
    print("=" * 50)
    
    for path in possible_paths:
        print(f"Checking: {path}")
        if os.path.exists(path):
            print(f"✅ File exists: {path}")
            print(f"   File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
            
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Model loaded successfully from: {path}")
                print(f"   Model summary:")
                model.summary()
                return True
            except Exception as e:
                print(f"❌ Error loading model from {path}: {str(e)}")
        else:
            print(f"❌ File not found: {path}")
        print("-" * 30)
    
    return False

if __name__ == "__main__":
    success = test_model_paths()
    if success:
        print("\n🎉 Model loading test completed successfully!")
    else:
        print("\n❌ Model loading test failed!") 