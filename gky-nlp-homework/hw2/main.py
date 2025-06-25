from src.data_loader import load_dataset
from src.preprocessing import preprocess
from src.vectorization import vectorize_bow, vectorize_tfidf
from src.sentiment_analysis import analyze_sentiment
from src.topic_modeling import prepare_bow, apply_lda, display_topics, save_topics_to_file

import os
import pandas as pd

def main():
    print("Dataset is loading...")
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "News_Category_Dataset_v3.json")
    df = load_dataset(DATA_PATH)
    
    print("Preprocessing first 500 data")
    df = df.head(500)  
    df["clean"] = df["headline"].apply(preprocess)

    print("Starting sentiment analysis")
    df["sentiment"] = df["clean"].apply(analyze_sentiment)

    print("BoW ve TF-IDF vectorizing")
    X_bow, bow_vectorizer = vectorize_bow(df["clean"])
    X_tfidf, tfidf_vectorizer = vectorize_tfidf(df["clean"])

    print("First 10 output:")
    print(df[["headline", "clean", "sentiment"]].head(10))

    print(f"BoW size: {X_bow.shape}")
    print(f"TF-IDF size: {X_tfidf.shape}")

    print("\n Topic Modeling starting...")
    X_lda, lda_vectorizer = prepare_bow(df["clean"])
    lda_model = apply_lda(X_lda, n_topics=5)
    #display_topics(lda_model, lda_vectorizer.get_feature_names_out(), no_top_words=8)
    #save topic
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(BASE_DIR, "results", "topic_keywords.txt")
    save_topics_to_file(
    lda_model,
    lda_vectorizer.get_feature_names_out(),
    output_path,
    no_top_words=8
)

if __name__ == "__main__":
    main()
