#!/usr/bin/env python3
"""
CNN/DailyMail Otomatik Özetleme Demo
Hızlı test için basit demo script'i
"""

import pandas as pd
from text_summarization import TextSummarizer
import torch

def quick_demo():
    """
    Hızlı demo fonksiyonu
    """
    print("CNN/DailyMail Otomatik Özetleme Demo")
    print("=" * 50)
    
    # GPU kontrolü
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")
    
    # Veri setini yükle (sadece birkaç örnek)
    print("\nVeri yükleniyor...")
    try:
        df = pd.read_csv('train.csv', nrows=10)
        print(f"Yüklendi: {len(df)} örnek")
    except FileNotFoundError:
        print("Veri dosyası bulunamadı!")
        print("Lütfen train.csv dosyasının mevcut olduğundan emin olun.")
        return
    
    # Özetleyici oluştur
    print("\nModel yükleniyor...")
    summarizer = TextSummarizer()
    
    # Veriyi ön işle
    print("\nVeri ön işleme...")
    processed_df = summarizer.preprocess_data(df, sample_size=5)
    
    # Örnek özetler oluştur
    print("\nÖrnek özetler oluşturuluyor...")
    print("=" * 80)
    
    for i in range(min(3, len(processed_df))):
        article = processed_df.iloc[i]['article_clean']
        reference = processed_df.iloc[i]['highlights_clean']
        
        print(f"\nÖrnek {i+1}:")
        print(f"Orijinal Metin (ilk 150 karakter):")
        print(f"  {article[:150]}...")
        print(f"\nGerçek Özet:")
        print(f"  {reference}")
        
        # Model özeti oluştur
        try:
            prediction = summarizer.generate_summary(article)
            print(f"\nModel Özeti:")
            print(f"  {prediction}")
        except Exception as e:
            print(f"\nHata: {e}")
        
        print("-" * 80)
    
    print("\nDemo tamamlandı!")
    print("\nTam eğitim için 'python text_summarization.py' komutunu çalıştırın.")

if __name__ == "__main__":
    quick_demo() 