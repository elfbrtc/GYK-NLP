# CNN/DailyMail Otomatik Özetleme Sistemi

Bu proje, CNN/DailyMail veri seti kullanarak T5-small transformer modeli ile otomatik özetleme sistemi geliştirmektedir.

## 📋 Proje Açıklaması

Bu sistem, haber metinlerinden otomatik olarak özgün özetler çıkaran bir transformer tabanlı modeldir. T5-small modeli kullanılarak article alanından summary alanını tahmin eden bir model eğitilmiştir.

## 🎯 Hedefler

- Transformer mimarisi kullanarak otomatik özetleme
- CNN/DailyMail veri seti üzerinde eğitim
- ROUGE skorları ile değerlendirme
- Pratik ve kullanılabilir bir sistem

## 📊 Veri Seti

**CNN/DailyMail Veri Seti:**
- **Train:** ~287,000 örnek
- **Validation:** ~13,000 örnek  
- **Test:** ~11,000 örnek

**Sütunlar:**
- `id`: Benzersiz kimlik
- `article`: Haber metni
- `highlights`: Özet (hedef)

## 🛠️ Teknik Detaylar

### Model
- **Model:** T5-small (60M parametre)
- **Mimari:** Transformer Encoder-Decoder
- **Dil:** İngilizce

### Hiperparametreler
- **Learning Rate:** 3e-5
- **Batch Size:** 8
- **Epochs:** 3
- **Max Input Length:** 512 token
- **Max Target Length:** 128 token
- **Optimizer:** AdamW
- **Weight Decay:** 0.01

### Veri Ön İşleme
- Metin temizleme (küçük harfe çevirme, özel karakter temizleme)
- Sequence truncation ve padding
- Prefix ekleme ("summarize: ")

## 📁 Dosya Yapısı

```
hw5/
├── text_summarization.py          # Ana Python script
├── text_summarization_notebook.ipynb  # Jupyter Notebook
├── requirements.txt               # Gerekli kütüphaneler
├── README.md                      # Bu dosya
├── train.csv                      # Eğitim verisi
├── validation.csv                 # Doğrulama verisi
├── test.csv                       # Test verisi
├── results.json                   # Sonuçlar (eğitim sonrası)
├── saved_model/                   # Kaydedilen model (eğitim sonrası)
└── results/                       # Eğitim sonuçları (eğitim sonrası)
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Python Script ile Çalıştırma

```bash
python text_summarization.py
```

### 3. Jupyter Notebook ile Çalıştırma

```bash
jupyter notebook text_summarization_notebook.ipynb
```

## 📈 Sonuçlar

### ROUGE Skorları
- **ROUGE-1:** ~0.35-0.40
- **ROUGE-2:** ~0.15-0.20  
- **ROUGE-L:** ~0.30-0.35

### Örnek Çıktılar
Sistem, eğitim sonrasında aşağıdaki gibi özetler üretir:

**Orijinal Metin:** "By . Associated Press . PUBLISHED: . 14:11 EST..."

**Gerçek Özet:** "Bishop John Folda, of North Dakota, is taking..."

**Tahmin Edilen Özet:** "Bishop John Folda of North Dakota is taking..."

## 🔧 Özelleştirme

### Model Değiştirme
```python
# BART modeli kullanmak için
summarizer = TextSummarizer(model_name="facebook/bart-base")
```

### Hiperparametre Ayarlama
```python
# Eğitim parametrelerini değiştirmek için
trainer = summarizer.train_model(
    train_dataset, 
    validation_dataset, 
    num_epochs=5,  # Epoch sayısını artır
    batch_size=16  # Batch size'ı artır
)
```

### Veri Boyutu Ayarlama
```python
# Daha fazla veri kullanmak için
train_sample = summarizer.preprocess_data(train_df, sample_size=5000)
```

## 📊 Değerlendirme Metrikleri

### ROUGE Skorları
- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap  
- **ROUGE-L:** Longest Common Subsequence

### İnsan Değerlendirmesi
- Anlamlılık
- Tutarlılık
- Kısalık

## 🎓 Geliştirme Süreci

### 1. Veri Ön İşleme
- Metin temizleme ve normalizasyon
- Uzunluk sınırlamaları
- Tokenization

### 2. Model Seçimi
- T5-small: Düşük donanım gereksinimi
- Pre-trained model avantajı
- Sequence-to-sequence mimari

### 3. Eğitim Stratejisi
- Transfer learning
- Fine-tuning
- Validation monitoring

### 4. Optimizasyon
- Learning rate tuning
- Batch size optimization
- Early stopping

## 🔍 Gelecek Geliştirmeler

- [ ] Daha büyük model kullanımı (T5-base, T5-large)
- [ ] Data augmentation teknikleri
- [ ] Ensemble methods
- [ ] Türkçe dil desteği
- [ ] Web arayüzü

## 📚 Referanslar

- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ROUGE Metric](https://en.wikipedia.org/wiki/ROUGE_(metric))
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)

## 👨‍💻 Geliştirici

Bu proje NLP dersi kapsamında geliştirilmiştir.

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. 