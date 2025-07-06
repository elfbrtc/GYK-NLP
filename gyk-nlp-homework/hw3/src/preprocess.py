
import re
import string
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def prepare_text(df, max_len=50):
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def encode_labels(label_lists):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_lists)
    return y, mlb

def tokenize_and_pad(X_train, X_test, max_len=50):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    return X_train_pad, X_test_pad, tokenizer