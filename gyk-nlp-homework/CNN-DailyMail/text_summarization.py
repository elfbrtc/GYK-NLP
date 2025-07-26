# =============================================================================
# CNN/DailyMail Otomatik Metin Özetleme Projesi
# =============================================================================
# Bu proje, CNN/DailyMail veri setini kullanarak T5 modeli ile otomatik metin özetleme
# yapan bir sistemdir. Proje şu ana bileşenlerden oluşur:
# 1. Veri ön işleme ve temizleme
# 2. T5 modeli ile fine-tuning
# 3. ROUGE metrikleri ile değerlendirme
# 4. Özet üretimi ve sonuç analizi
# =============================================================================

import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# NLTK verilerini indir - Cümle tokenizasyonu için gerekli
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSummarizer:
    """
    Ana sınıf: Metin özetleme işlemlerini yönetir
    Bu sınıf T5 modelini kullanarak metin özetleme yapar
    """
    def __init__(self, model_name="t5-small", max_input_length=512, max_target_length=128):
        """
        Text Summarizer sınıfının başlatıcısı
        
        Args:
            model_name: Kullanılacak T5 model versiyonu (t5-small, t5-base, t5-large)
            max_input_length: Giriş metninin maksimum token sayısı
            max_target_length: Özet metninin maksimum token sayısı
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Hugging Face'den T5 modelini ve tokenizer'ını yükle
        print(f"Model yükleniyor: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # ROUGE skorlarını hesaplamak için scorer oluştur
        # ROUGE: Özet kalitesini değerlendiren standart metrik
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def clean_text(self, text):
        """
        Metni temizleme fonksiyonu
        - Küçük harfe çevirme
        - Gereksiz boşlukları temizleme
        - Özel karakterleri kaldırma
        - NaN değerleri kontrol etme
        """
        if pd.isna(text):
            return ""
        
        # Küçük harfe çevir - T5 modeli case-sensitive değil
        text = text.lower()
        
        # Birden fazla boşluğu tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        
        # Noktalama işaretleri dışındaki özel karakterleri temizle
        # Noktalama işaretleri cümle yapısı için önemli
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        return text.strip()
    
    def preprocess_data(self, df, sample_size=None):
        """
        Veri setini ön işleme fonksiyonu
        - Veri temizleme
        - Örnek sayısını sınırlama (hızlı prototip için)
        - Çok kısa metinleri filtreleme
        """
        print("Veri ön işleme başlıyor...")
        
        # Eğer sample_size belirtilmişse, rastgele örnek al
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)  # Tekrarlanabilirlik için seed
        
        # Article ve highlights sütunlarını temizle
        df['article_clean'] = df['article'].apply(self.clean_text)
        df['highlights_clean'] = df['highlights'].apply(self.clean_text)
        
        # Çok kısa metinleri filtrele (anlamlı özet için yeterli içerik olmalı)
        df = df[(df['article_clean'].str.len() > 50) & (df['highlights_clean'].str.len() > 10)]
        
        print(f"Ön işleme sonrası veri sayısı: {len(df)}")
        return df
    
    def tokenize_function(self, examples):
        """
        Veriyi T5 modeli için tokenize etme fonksiyonu
        - Giriş metnine "summarize:" prefix'i ekleme
        - Padding ve truncation işlemleri
        - Label'ları hazırlama
        """
        # T5 için özel prefix: "summarize:" komutu modelin özetleme yapacağını belirtir
        inputs = ["summarize: " + article for article in examples["article_clean"]]
        
        # Giriş metinlerini tokenize et
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,  # Maksimum uzunluk sınırı
            padding="max_length",              # Kısa metinleri padding ile uzat
            truncation=True,                   # Uzun metinleri kes
            return_tensors="pt"                # PyTorch tensor formatında döndür
        )
        
        # Hedef metinleri (özetleri) tokenize et
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["highlights_clean"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # Model girişlerine label'ları ekle
        model_inputs["labels"] = labels["input_ids"]
        
        # Padding token'larını -100 ile değiştir (loss hesaplamasında göz ardı edilir)
        model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100
        
        return model_inputs
    
    def train_model(self, train_dataset, eval_dataset, num_epochs=3, batch_size=8):
        """
        Modeli eğitme fonksiyonu
        - Seq2SeqTrainer kullanarak fine-tuning
        - Validation seti ile değerlendirme
        - Checkpoint kaydetme
        """
        print("Model eğitimi başlıyor...")
        
        # Eğitim parametrelerini ayarla
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",           # Sonuçları kaydetme dizini
            eval_strategy="epoch",            # Her epoch sonunda değerlendir
            save_strategy="epoch",            # Her epoch sonunda modeli kaydet
            learning_rate=3e-5,               # Öğrenme oranı (küçük değer fine-tuning için)
            per_device_train_batch_size=batch_size,    # GPU başına batch boyutu
            per_device_eval_batch_size=batch_size,     # Değerlendirme batch boyutu
            weight_decay=0.01,                # Regularization için weight decay
            save_total_limit=3,               # Maksimum 3 checkpoint sakla
            num_train_epochs=num_epochs,      # Toplam epoch sayısı
            predict_with_generate=True,       # Özet üretimi için gerekli
            fp16=torch.cuda.is_available(),   # GPU varsa mixed precision kullan
            logging_steps=100,                # Her 100 adımda log al
            warmup_steps=500,                 # İlk 500 adımda learning rate'i artır
            load_best_model_at_end=True,      # En iyi modeli yükle
            metric_for_best_model="eval_loss", # En iyi modeli belirleme kriteri
            greater_is_better=False,          # Loss için düşük değer daha iyi
            report_to=[],                     # TensorBoard'u devre dışı bırak
        )
        
        # Seq2SeqTrainer oluştur (sequence-to-sequence modeller için özel trainer)
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Modeli eğit
        trainer.train()
        
        return trainer
    
    def generate_summary(self, text, max_length=128):
        """
        Tek bir metin için özet oluşturma fonksiyonu
        - Giriş metnini temizleme
        - Tokenization
        - Beam search ile özet üretimi
        - Decoding
        """
        # Metni temizle
        clean_text = self.clean_text(text)
        
        # T5 için gerekli prefix'i ekle
        input_text = "summarize: " + clean_text
        
        # Metni tokenize et
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # GPU'ya taşı (varsa)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Özet oluştur (inference modunda)
        with torch.no_grad():  # Gradient hesaplamayı kapat (hız için)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,        # Maksimum özet uzunluğu
                num_beams=4,                  # Beam search için beam sayısı
                length_penalty=2.0,           # Uzun özetleri tercih et
                early_stopping=True,          # EOS token'ı görünce dur
                no_repeat_ngram_size=3        # 3-gram tekrarını engelle
            )
        
        # Token'ları metne çevir
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def evaluate_rouge(self, predictions, references):
        """
        ROUGE skorlarını hesaplama fonksiyonu
        ROUGE: Recall-Oriented Understudy for Gisting Evaluation
        - ROUGE-1: Tek kelime örtüşmesi
        - ROUGE-2: İki kelime örtüşmesi  
        - ROUGE-L: En uzun ortak alt dizi
        """
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        # Her tahmin-gerçek çifti için ROUGE skorlarını hesapla
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Ortalama skorları döndür
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def save_model(self, path="./saved_model"):
        """
        Eğitilmiş modeli kaydetme fonksiyonu
        - Modeli ve tokenizer'ı belirtilen dizine kaydet
        - Daha sonra yüklemek için
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model kaydedildi: {path}")
    
    def load_model(self, path="./saved_model"):
        """
        Kaydedilmiş modeli yükleme fonksiyonu
        - Önceden eğitilmiş modeli yükle
        - Yeni eğitim yapmadan kullanmak için
        """
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        print(f"Model yüklendi: {path}")

def main():
    """
    Ana fonksiyon: Tüm işlem akışını yönetir
    Bu fonksiyon projenin ana giriş noktasıdır
    """
    print("CNN/DailyMail Otomatik Özetleme Sistemi")
    print("=" * 50)
    
    # =============================================================================
    # 1. VERİ YÜKLEME AŞAMASI
    # =============================================================================
    print("Veri setleri yükleniyor...")
    train_df = pd.read_csv('train.csv')      # Eğitim verisi
    validation_df = pd.read_csv('validation.csv')  # Doğrulama verisi
    test_df = pd.read_csv('test.csv')        # Test verisi
    
    print(f"Train seti: {len(train_df)} örnek")
    print(f"Validation seti: {len(validation_df)} örnek")
    print(f"Test seti: {len(test_df)} örnek")
    
    # =============================================================================
    # 2. MODEL VE ÖN İŞLEME AŞAMASI
    # =============================================================================
    # TextSummarizer sınıfının bir örneğini oluştur
    summarizer = TextSummarizer()
    
    # Veriyi ön işle (hızlı prototip için küçük örnek)
    print("\nVeri ön işleme...")
    train_sample = summarizer.preprocess_data(train_df, sample_size=1000)    # 1000 örnek
    validation_sample = summarizer.preprocess_data(validation_df, sample_size=200)  # 200 örnek
    test_sample = summarizer.preprocess_data(test_df, sample_size=200)       # 200 örnek
    
    # Pandas DataFrame'leri Hugging Face Dataset formatına çevir
    train_dataset = Dataset.from_pandas(train_sample)
    validation_dataset = Dataset.from_pandas(validation_sample)
    test_dataset = Dataset.from_pandas(test_sample)
    
    # =============================================================================
    # 3. TOKENIZATION AŞAMASI
    # =============================================================================
    print("Veri tokenize ediliyor...")
    # Her veri setini T5 modeli için uygun formata çevir
    train_dataset = train_dataset.map(summarizer.tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(summarizer.tokenize_function, batched=True)
    test_dataset = test_dataset.map(summarizer.tokenize_function, batched=True)
    
    # =============================================================================
    # 4. MODEL EĞİTİM AŞAMASI
    # =============================================================================
    print("\nModel eğitimi başlıyor...")
    # Seq2SeqTrainer ile modeli fine-tune et
    trainer = summarizer.train_model(train_dataset, validation_dataset, num_epochs=3)
    
    # =============================================================================
    # 5. DEĞERLENDİRME AŞAMASI
    # =============================================================================
    print("\nTest seti üzerinde değerlendirme...")
    test_predictions = []    # Model tahminleri
    test_references = []     # Gerçek özetler
    
    # Test setinden 50 örnek al ve özet oluştur
    for i in tqdm(range(min(50, len(test_sample)))):
        article = test_sample.iloc[i]['article_clean']      # Giriş metni
        reference = test_sample.iloc[i]['highlights_clean'] # Gerçek özet
        
        # Model ile özet oluştur
        prediction = summarizer.generate_summary(article)
        
        test_predictions.append(prediction)
        test_references.append(reference)
    
    # =============================================================================
    # 6. ROUGE SKORLARINI HESAPLAMA
    # =============================================================================
    # Özet kalitesini değerlendirmek için ROUGE metriklerini hesapla
    rouge_scores = summarizer.evaluate_rouge(test_predictions, test_references)
    
    print("\nROUGE Skorları:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")  # Tek kelime örtüşmesi
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")  # İki kelime örtüşmesi
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")  # En uzun ortak alt dizi
    
    # =============================================================================
    # 7. ÖRNEK ÇIKTILARI GÖSTERME
    # =============================================================================
    print("\nÖrnek Özetler:")
    print("=" * 80)
    
    # İlk 5 örneği göster
    for i in range(min(5, len(test_predictions))):
        print(f"\nÖrnek {i+1}:")
        print(f"Orijinal Metin (ilk 200 karakter): {test_sample.iloc[i]['article_clean'][:200]}...")
        print(f"Gerçek Özet: {test_references[i]}")
        print(f"Tahmin Edilen Özet: {test_predictions[i]}")
        print("-" * 80)
    
    # =============================================================================
    # 8. MODEL VE SONUÇLARI KAYDETME
    # =============================================================================
    # Eğitilmiş modeli kaydet (daha sonra kullanmak için)
    summarizer.save_model()
    
    # Sonuçları JSON formatında kaydet
    results = {
        'rouge_scores': rouge_scores,
        'examples': []
    }
    
    # Örnek çıktıları da kaydet
    for i in range(min(5, len(test_predictions))):
        results['examples'].append({
            'original_text': test_sample.iloc[i]['article_clean'][:200] + "...",
            'reference_summary': test_references[i],
            'predicted_summary': test_predictions[i]
        })
    
    # Sonuçları JSON dosyasına kaydet
    import json
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nSonuçlar 'results.json' dosyasına kaydedildi.")
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main() 