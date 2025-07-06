import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_merge_data(data_dir='data/full_dataset/'):
    files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
    dfs = []

    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} can't be found.")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    emotion_columns = df.columns[10:]
    labels_df = df[emotion_columns].astype(int)
    df_text = df[['text']].copy()

    #If no split column create and split with sklearn
    X_temp, X_test, y_temp, y_test = train_test_split(
        df_text, labels_df, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42)  # 0.1 of 0.8 = 0.08

    #Add split column
    X_train['split'] = 'train'
    X_val['split'] = 'val'
    X_test['split'] = 'test'

    df_split = pd.concat([X_train, X_val, X_test])
    labels_split = pd.concat([y_train, y_val, y_test])

    return df_split.reset_index(drop=True), labels_split.reset_index(drop=True)
