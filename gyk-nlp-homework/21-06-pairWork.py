# N-gram Modeller , Word Embeddings

# N-gram Modeller

# Bir metindeki kelimelerin (ya da karakterlerin) ardışık gruplar halinde oluşturulmasıdır.

# NLP çok eğlenceli alan
# Unigram-Bigram-Trigram 


# Unigram (1n) = ["NLP", "çok", "eğlenceli", "alan"]
# Bigram (2n) = ["NLP çok", "çok eğlenceli", "eğlenceli alan"]
# Trigram (3n) = ["NLP çok eğlenceli", "çok eğlenceli alan"]

# Otomatik tamamlama, spam tespiti, yazım önerisi.
# Nerede kullanılır? => Dilin anlamını anlamaz. Sadece istatistiksel olarak kullanılır.

# Apple is a fruit.
# Apple is a company.

corpus = [
    "NLP çok eğlenceli alan",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz"
]

from sklearn.feature_extraction.text import TfidfVectorizer
# Unigram ve Bigram birlikte kullanılır.
vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)

X = vectorizer.fit_transform(corpus)

print(f"Feature Names: {vectorizer.get_feature_names_out()}")
print(f"X: {X.toarray()}")

# Word Embedding
# Her kelimeye sayısal bir vektör ata. Bu vektörler sayesinde:
# Kelimeler arasındaki anlamsal yakınlık öğreniliyor.
# Aynı bağlam geçen kelimeler, uzayda da birbirine yakın olur.

# Araba -> [0.21, -0.43, 0.92, ........, 0.01] 100 veya 300+ boyutlu.

# Güzel ek özellik => Vektör cebiri bile yapılabilir.
# vec("king") - vec("man") + vec("woman") = vec("queen")

# Nerede kullanılır? 

# Derin öğrenme.
# Chatbot, anlamsal arama

corpus = [
    "NLP çok eğlenceli alan",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz"
]
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

#her cümleyi tokenize et. kelime listesi oluştur.
# kelimeleri parçala, liste haline getir.
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
print("******")
print(tokenized_sentences)

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)

print("*******")
print(model.wv['nlp'])
print("*******")
print(model.wv.most_similar('nlp'))


# Sentence Embedding


corpus = [
    "NLP çok eğlenceli alan",
    "NLP çok önemli",
    "Eğlenceli projeler yapıyoruz"
]
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)

# Ortalama Vektör Alma
import numpy as np

def sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    return np.zeros(100)

vec1 = sentence_vector(corpus[0])
vec2 = sentence_vector(corpus[1])

print(vec1)
print(vec2)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# cosinesimilarty(a,b) = a.b / |a| * |b| => -1,1 arasında değer döner.
#
# Average Word Embedding

print(cosine_similarity(vec1, vec2))



# Sentence-BERT (SBERT) 

from sentence_transformers import SentenceTransformer, util

# Modeli yükle - Türkçe için 'paraphrase-multilingual-MiniLM-L12-v2' 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

sentences = [
    "NLP çok eğlenceli alan",
    "NLP çok önemli",
    "Eğlenceli projeler yapıyoruz"
]

# Cümleleri vektörleştir
embeddings = model.encode(sentences)

# İki cümle arasındaki benzerlik
#cos_sim = util.cos_sim(embeddings[0], embeddings[1])
cos_sim = cosine_similarity(embeddings[0], embeddings[1])
print(f"Benzerlik: {cos_sim.item()}")