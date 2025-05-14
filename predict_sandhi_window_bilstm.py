from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Bidirectional, Embedding, Reshape, Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import train_test_data_prepare as sdp

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

def train_predict_sandhi_window(dtrain, dtest, mode):
    # Configuration
    batch_size = 64  # Batch size for training
    epochs = 40  # Number of epochs to train for
    latent_dim = 64  # Latent dimensionality of the encoding space
    inwordlen = 5
    model_dir = "models"
    model_path = os.path.join(model_dir, "bilstm.h5")
    char_index_path = os.path.join(model_dir, "bilstm_char_index.pkl")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Vectorize the data
    inputs = []
    targets = []
    characters = set()
    
    print(f"Processing {len(dtrain)} training examples")
    
    for data in dtrain:
        # Ensure data has correct format
        if len(data) < 6:
            print(f"Warning: Invalid data format: {data}")
            continue
            
        target = np.zeros(len(data[3]))
        input_word = data[3]
    
        inputs.append(input_word)
        for i in range(data[4], data[5]):
            target[i] = 1
        targets.append(target)
    
        for char in input_word:
            if char not in characters:
                characters.add(char)
    
    # Add test data characters to ensure consistency
    for data in dtest:
        if len(data) < 6:
            print(f"Warning: Invalid test data format: {data}")
            continue
            
        for char in data[3]:
            if char not in characters:
                characters.add(char)

    # Get maximum sequence length
    maxlen = max([len(s) for s in inputs])
    print(f"Maximum sequence length: {maxlen}")

    # Add padding character
    characters.add('*')
    
    # Create character to index mapping
    char2idx = dict([(char, i) for i, char in enumerate(characters)])
    num_tokens = len(characters)
    
    # Save character mapping for inference
    save_model_data(char2idx, char_index_path)
    
    # Prepare training data
    X_train = [[char2idx[c] for c in w] for w in inputs]
    X_train = pad_sequences(maxlen=maxlen, sequences=X_train, padding="post", value=char2idx['*'])
    
    Y_train = targets
    Y_train = pad_sequences(maxlen=maxlen, sequences=Y_train, padding="post", value=0.0)
    Y_train = np.array(Y_train).reshape(-1, maxlen, 1)
    
    # Prepare test data
    inputs_test = []
    targets_test = []
    
    for data in dtest:
        if len(data) < 6:
            continue
            
        target = np.zeros(len(data[3]))
        input_word = data[3]
    
        inputs_test.append(input_word)
        for i in range(data[4], data[5]):
            target[i] = 1
        targets_test.append(target)
    
    X_test = []
    for w in inputs_test:
        # Handle characters not seen in training
        seq = []
        for c in w:
            if c in char2idx:
                seq.append(char2idx[c])
            else:
                # Use a default index for unknown characters
                seq.append(char2idx['*'])
        X_test.append(seq)
    
    X_test = pad_sequences(maxlen=maxlen, sequences=X_test, padding="post", value=char2idx['*'])
    
    Y_test = targets_test
    Y_test = pad_sequences(maxlen=maxlen, sequences=Y_test, padding="post", value=0.0)
    Y_test = np.array(Y_test).reshape(-1, maxlen, 1)
    
    print('Number of training samples:', len(X_train))
    print('Number of test samples:', len(X_test))
    print('Number of unique tokens:', num_tokens)
    
    # Check if model exists and should be loaded
    if os.path.exists(model_path) and mode == 1:
        print(f"Loading existing model from {model_path}")
        model = load_model(model_path)
    else:
        print("Training new model")
        # Define an input sequence and process it
        inputword = Input(shape=(maxlen,))
        embed = Embedding(input_dim=num_tokens, output_dim=8, input_length=maxlen, mask_zero=True)(inputword)
        bilstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
        out, forward_h, forward_c, backward_h, backward_c = bilstm(embed)
        outd = Dropout(0.5)(out)
        outputtarget = Dense(1, activation="sigmoid")(outd)
        
        model = Model(inputword, outputtarget)
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
        model.summary()
        model.fit(X_train, Y_train, batch_size, epochs, validation_split=0.1)
        
        # Save model
        model.save(model_path)
    
    # Save test data for later evaluation
    np.save(os.path.join(model_dir, 'testX'), X_test)
    np.save(os.path.join(model_dir, 'testY'), Y_test)
    
    np.set_printoptions(precision=2, suppress=True)
    passed = 0
    failed = 0
   
    startlist = []
    for i in range(X_test.shape[0]):
        test = X_test[i].reshape((-1, maxlen))
        res = model.predict(test, verbose=0)  # Suppress prediction verbose output
        res = res.reshape((maxlen))
        dup = np.copy(res)
        act = Y_test[i].reshape((maxlen))
    
        # Find the window with highest probability sum
        maxsum = 0
        maxstart = 0
        for j in range(maxlen-inwordlen):
            sumword = 0
            for k in range(inwordlen):
                sumword = sumword + dup[j+k]
            if maxsum < sumword:
                maxsum = sumword
                maxstart = j

        startlist.append(maxstart)

        # Evaluate accuracy if in evaluation mode
        if mode == 0:
            actual_start = -1
            for k in range(len(act)):
                if act[k] == 1:
                    actual_start = k
                    break
    
            if actual_start == maxstart:
                passed = passed + 1
            else:
                failed = failed + 1

    # Print evaluation results if in evaluation mode
    if mode == 0:
        print(f"Correct window predictions: {passed}")
        print(f"Incorrect window predictions: {failed}")
        total = passed + failed
        accuracy = passed * 100 / total if total > 0 else 0
        print(f"Window prediction accuracy: {accuracy:.2f}%")

    return startlist

# For standalone testing
if __name__ == "__main__":
    dl = sdp.get_xy_data("Data/sandhiword.txt")
    dtrain, dtest = train_test_split(dl, test_size=0.2, random_state=1)
    train_predict_sandhi_window(dtrain, dtest, 0)
