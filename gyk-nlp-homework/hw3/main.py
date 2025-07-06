from src import (
    load_and_merge_data,
    prepare_text,
    tokenize_and_pad,
    create_model,
    train_model,
    evaluate_model
)

import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50
MODEL_PATH = "models/goemotions_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Load text and labels
df, y = load_and_merge_data()
df = prepare_text(df)

# Split data
X_train = df[df['split'] == 'train']['clean_text']
X_test = df[df['split'] == 'test']['clean_text']
y_train = y[df['split'] == 'train'].values
y_test = y[df['split'] == 'test'].values

# Load model & tokenizer if already saved
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    print("Saved model and tokenizer found. Loading...")
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Tokenize test data
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

else:
    print("No saved model found. Training a new one...")

    # Tokenize and pad both train and test
    X_train_pad, X_test_pad, tokenizer = tokenize_and_pad(X_train, X_test, max_len=MAX_LEN)

    # Create and train model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size, MAX_LEN, num_labels=y.shape[1])
    history = train_model(model, X_train_pad, y_train, X_test_pad, y_test, epochs=10)

    # Save model and tokenizer
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

# Evaluate model
print("\nEvaluating model on test set...")
y_pred = model.predict(X_test_pad, batch_size=32)
evaluate_model(y_test, y_pred)
