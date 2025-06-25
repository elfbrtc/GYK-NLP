from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import os

def prepare_bow(corpus, max_features=1000):
    """
    return BoW matrix and vectorizer
    """
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def apply_lda(X, n_topics=5):
    """
    train LDA model over BoW matrix
    """
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    return lda

def display_topics(model, feature_names, no_top_words=10):
    """
    show the significant words for each topic

    """
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(", ".join(top_words))

def save_topics_to_file(model, feature_names, output_path, no_top_words=10):
    """
    write topic keywords into the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            f.write(f"Topic {topic_idx + 1}:\n")
            f.write(", ".join(top_words) + "\n\n")
    print(f"Topics saved to {output_path}")

