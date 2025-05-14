# Fixed version of sandhi_vicceda.py with improved data processing and model handling
from train_test_data_prepare import get_xy_data
from predict_sandhi_window_bilstm import train_predict_sandhi_window
from split_sandhi_window_seq2seq_bilstm import train_sandhi_split
from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np

def save_model_data(data, filename):
    """Save model data for later use"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_model_data(filename):
    """Load model data from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Configuration
inwordlen = 5
model_dir = "models"
character_set_file = os.path.join(model_dir, "character_set.pkl")
token_index_file = os.path.join(model_dir, "token_index.pkl")

# Create model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

def run_sandhi_pipeline(training_file, test_file=None, force_retrain=False):
    """
    Run the complete sandhi pipeline with improved error handling
    and data preprocessing
    """
    print(f"Loading training data from {training_file}")
    
    # Get training data
    dl_train = get_xy_data(training_file)
    
    if not dl_train:
        print("Error: No valid training data found")
        return
    
    print(f"Found {len(dl_train)} valid training examples")
    
    # Get test data if provided
    if test_file:
        print(f"Loading test data from {test_file}")
        dl_test = get_xy_data(test_file)
        if not dl_test:
            print("Error: No valid test data found")
            return
        print(f"Found {len(dl_test)} valid test examples")
    else:
        # Split the training data
        dl_train, dl_test = train_test_split(dl_train, test_size=0.2, random_state=1)
    
    # Validate data structure
    for data in dl_train + dl_test:
        if len(data) != 8:
            print(f"Error: Invalid data format. Expected 8 elements per example, got {len(data)}")
            return
    
    print("Training sandhi window prediction model...")
    
    # Predict the sandhi window
    try:
        sl = train_predict_sandhi_window(dl_train, dl_test, 1)
    except Exception as e:
        print(f"Error in sandhi window prediction: {e}")
        return
    
    if len(sl) != len(dl_test):
        print(f"Error: Mismatch in results. Expected {len(dl_test)}, got {len(sl)}")
        return
    
    # Update test data with predicted sandhi windows
    for i in range(len(dl_test)):
        start = sl[i]
        end = sl[i] + inwordlen
        flen = len(dl_test[i][3])
        if end > flen:
            end = flen
        dl_test[i][2] = dl_test[i][3][start:end]
        dl_test[i][4] = start
        dl_test[i][5] = end
    
    print("Training sandhi split model...")
    
    # Split the sandhi
    try:
        results = train_sandhi_split(dl_train, dl_test, 1)
    except Exception as e:
        print(f"Error in sandhi splitting: {e}")
        return
    
    if len(results) != len(dl_test):
        print(f"Error: Mismatch in results. Expected {len(dl_test)}, got {len(results)}")
        return
    
    # Evaluate the results
    passed = 0
    failed = 0
    
    for i in range(len(dl_test)):
        start = dl_test[i][4]
        end = dl_test[i][5]
        splitword = dl_test[i][3][:start] + results[i] + dl_test[i][3][end:]
        actword = dl_test[i][6] + '+' + dl_test[i][7]
        
        if splitword == actword:
            passed += 1
        else:
            failed += 1
            if failed < 10:  # Show some examples of failures for debugging
                print(f"Failed example {i}:")
                print(f"Original: {dl_test[i][3]}")
                print(f"Predicted split: {splitword}")
                print(f"Actual split: {actword}")
                print(f"Window: {dl_test[i][2]} at position {start}:{end}")
                print("-" * 50)
    
    total = passed + failed
    accuracy = passed * 100 / total if total > 0 else 0
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    # Example usage with literature dataset
    training_file = "Data/SandhiKosh/literature_train.txt"
    test_file = "Data/SandhiKosh/literature_test.txt"
    
    print("Running sandhi pipeline with separate test set")
    run_sandhi_pipeline(training_file, test_file)