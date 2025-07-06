# Fonkisyon 

# pipeline => 
# 1-Tokenization - lowercasing 
# 2- Stopwords Temizliği
# 3- Lemmatization
# 4- TF-IDF Vektörleştirme
# 5- Feature isimlerini ve arrayi ekrana yazdır.

# generate a corpus of 10 about AI in english
# corpus = [
#     "Artificial Intelligence is the future.",
#     "AI is changing the world.",
#     "AI is a branch of computer science.",
# ]

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Gerekli NLTK verilerini indiriyoruz (ilk seferde çalıştır)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def nlp_pipeline(corpus):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_corpus = []

    for text in corpus:
        # 1- Lowercase
        text = text.lower()

        # 2- Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)  # noktalama kaldır
        text = re.sub(r'\d+', '', text)      # rakamları kaldır

        # 3- Tokenization
        tokens = word_tokenize(text)

        # 4- Stopword temizliği
        tokens = [word for word in tokens if word not in stop_words]

        # 5- Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Tekrar metin haline getir
        cleaned_text = ' '.join(lemmatized_tokens)
        cleaned_corpus.append(cleaned_text)

    # 6- TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_corpus)

    # Feature isimleri
    features = vectorizer.get_feature_names_out()
    print("Features (Kelime Listesi):")
    print(features)

    # TF-IDF vektör matrisi (array olarak)
    print("\nTF-IDF Vektör Matrisi:")
    print(X.toarray())

# Test corpus
corpus = [
    "Artificial Intelligence is the future.",
    "AI is changing the world.",
    "AI is a branch of computer science.",
]

nlp_pipeline(corpus)
