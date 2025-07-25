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

# NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSummarizer:
    def __init__(self, model_name="t5-small", max_input_length=512, max_target_length=128):
        """
        Text Summarizer sınıfı
        
        Args:
            model_name: Kullanılacak model adı
            max_input_length: Maksimum giriş uzunluğu
            max_target_length: Maksimum hedef uzunluğu
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Tokenizer ve model yükle
        print(f"Model yükleniyor: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def clean_text(self, text):
        """
        Metni temizle
        """
        if pd.isna(text):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Gereksiz boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        return text.strip()
    
    def preprocess_data(self, df, sample_size=None):
        """
        Veriyi ön işle
        """
        print("Veri ön işleme başlıyor...")
        
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Metinleri temizle
        df['article_clean'] = df['article'].apply(self.clean_text)
        df['highlights_clean'] = df['highlights'].apply(self.clean_text)
        
        # Boş metinleri filtrele
        df = df[(df['article_clean'].str.len() > 50) & (df['highlights_clean'].str.len() > 10)]
        
        print(f"Ön işleme sonrası veri sayısı: {len(df)}")
        return df
    
    def tokenize_function(self, examples):
        """
        Veriyi tokenize et
        """
        # Giriş metni için prefix ekle
        inputs = ["summarize: " + article for article in examples["article_clean"]]
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Hedef metinleri tokenize et
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["highlights_clean"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        # Padding token'ları -100 ile değiştir
        model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100
        
        return model_inputs
    
    def train_model(self, train_dataset, eval_dataset, num_epochs=3, batch_size=8):
        """
        Modeli eğit
        """
        print("Model eğitimi başlıyor...")
        
        # Eğitim argümanları
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            warmup_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # TensorBoard'u devre dışı bırak
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Eğitim
        trainer.train()
        
        return trainer
    
    def generate_summary(self, text, max_length=128):
        """
        Tek bir metin için özet oluştur
        """
        # Metni temizle
        clean_text = self.clean_text(text)
        
        # Prefix ekle
        input_text = "summarize: " + clean_text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # GPU'ya taşı
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Özet oluştur
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def evaluate_rouge(self, predictions, references):
        """
        ROUGE skorlarını hesapla
        """
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def save_model(self, path="./saved_model"):
        """
        Modeli kaydet
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model kaydedildi: {path}")
    
    def load_model(self, path="./saved_model"):
        """
        Modeli yükle
        """
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        print(f"Model yüklendi: {path}")

def main():
    """
    Ana fonksiyon
    """
    print("CNN/DailyMail Otomatik Özetleme Sistemi")
    print("=" * 50)
    
    # Veri setlerini yükle
    print("Veri setleri yükleniyor...")
    train_df = pd.read_csv('train.csv')
    validation_df = pd.read_csv('validation.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Train seti: {len(train_df)} örnek")
    print(f"Validation seti: {len(validation_df)} örnek")
    print(f"Test seti: {len(test_df)} örnek")
    
    # Özetleyici oluştur
    summarizer = TextSummarizer()
    
    # Veriyi ön işle (hızlı prototip için küçük örnek)
    print("\nVeri ön işleme...")
    train_sample = summarizer.preprocess_data(train_df, sample_size=1000)
    validation_sample = summarizer.preprocess_data(validation_df, sample_size=200)
    test_sample = summarizer.preprocess_data(test_df, sample_size=200)
    
    # Dataset'e çevir
    train_dataset = Dataset.from_pandas(train_sample)
    validation_dataset = Dataset.from_pandas(validation_sample)
    test_dataset = Dataset.from_pandas(test_sample)
    
    # Tokenize et
    print("Veri tokenize ediliyor...")
    train_dataset = train_dataset.map(summarizer.tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(summarizer.tokenize_function, batched=True)
    test_dataset = test_dataset.map(summarizer.tokenize_function, batched=True)
    
    # Modeli eğit
    print("\nModel eğitimi başlıyor...")
    trainer = summarizer.train_model(train_dataset, validation_dataset, num_epochs=3)
    
    # Test seti üzerinde değerlendir
    print("\nTest seti üzerinde değerlendirme...")
    test_predictions = []
    test_references = []
    
    for i in tqdm(range(min(50, len(test_sample)))):
        article = test_sample.iloc[i]['article_clean']
        reference = test_sample.iloc[i]['highlights_clean']
        
        prediction = summarizer.generate_summary(article)
        
        test_predictions.append(prediction)
        test_references.append(reference)
    
    # ROUGE skorlarını hesapla
    rouge_scores = summarizer.evaluate_rouge(test_predictions, test_references)
    
    print("\nROUGE Skorları:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # Örnek çıktıları göster
    print("\nÖrnek Özetler:")
    print("=" * 80)
    
    for i in range(min(5, len(test_predictions))):
        print(f"\nÖrnek {i+1}:")
        print(f"Orijinal Metin (ilk 200 karakter): {test_sample.iloc[i]['article_clean'][:200]}...")
        print(f"Gerçek Özet: {test_references[i]}")
        print(f"Tahmin Edilen Özet: {test_predictions[i]}")
        print("-" * 80)
    
    # Modeli kaydet
    summarizer.save_model()
    
    # Sonuçları kaydet
    results = {
        'rouge_scores': rouge_scores,
        'examples': []
    }
    
    for i in range(min(5, len(test_predictions))):
        results['examples'].append({
            'original_text': test_sample.iloc[i]['article_clean'][:200] + "...",
            'reference_summary': test_references[i],
            'predicted_summary': test_predictions[i]
        })
    
    # Sonuçları JSON olarak kaydet
    import json
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nSonuçlar 'results.json' dosyasına kaydedildi.")
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main() 