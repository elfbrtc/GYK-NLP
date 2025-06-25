from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize_bow(corpus, max_features=1000):
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def vectorize_tfidf(corpus, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
