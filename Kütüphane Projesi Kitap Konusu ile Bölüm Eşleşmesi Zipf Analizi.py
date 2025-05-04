#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# Gerekli kütüphaneleri içe aktar
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# NLTK kaynaklarını indir
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Veri setini yükle
df = pd.read_csv('pg_catalog.csv', encoding='utf-8')

# Metin analizi için sütunlar
text_columns = ['Title', 'Authors', 'Subjects', 'Bookshelves']

# İngilizce stopwords listesi
english_stopwords = set(stopwords.words('english'))

# PorterStemmer nesnesi
ps = PorterStemmer()

# Metin ön işleme fonksiyonu
def preprocess_sentence(sentence):
    if not isinstance(sentence, str):
        return []
    # Tokenlaştırma
    tokens = word_tokenize(sentence)
    # Küçük harfe çevir, sadece alfabetik tokenları al, stopwords'leri kaldır
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in english_stopwords]
    # PorterStemmer ile stemleme
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    return stemmed_tokens

# Zipf analizi fonksiyonu
def perform_zipf_analysis(texts, column_name):
    # Tüm metinleri tokenlara ayır
    all_tokens = []
    for text in texts:
        tokens = preprocess_sentence(text)
        all_tokens.extend(tokens)
    
    # Kelime sıklıklarını hesapla
    word_counts = Counter(all_tokens)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Zipf analizi için sıralı sıklıkları ve sıraları al
    ranks = np.arange(1, len(sorted_word_counts) + 1)
    frequencies = [count for _, count in sorted_word_counts]
    
    # Log-log grafiğini çiz
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker='.', linestyle='none', label='Kelime Sıklıkları')
    plt.plot(ranks, frequencies[0] / ranks, 'r-', label='Zipf Kanunu (f ~ 1/r)')
    plt.xlabel('Sıra (Rank)')
    plt.ylabel('Frekans (Frequency)')
    plt.title(f'Zipf Yasası Analizi - {column_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    # İlk 10 kelimeyi yazdır
    print(f"\n{column_name} - En sık kullanılan ilk 10 kelime:")
    for word, count in sorted_word_counts[:10]:
        print(f"{word}: {count}")
    
    # Veri seti boyutunu değerlendir
    total_tokens = len(all_tokens)
    unique_tokens = len(word_counts)
    print(f"\n{column_name} - Veri Seti Değerlendirmesi:")
    print(f"Toplam token sayısı: {total_tokens}")
    print(f"Benzersiz token sayısı: {unique_tokens}")
    if total_tokens < 1000 or unique_tokens < 100:
        print("Uyarı: Veri seti boyutu sınırlı. Zipf Yasası'nın net gözlemlenmesi için daha büyük bir veri seti önerilir.")
    else:
        print("Veri seti boyutu, Zipf Yasası analizi için yeterli görünüyor.")

# Her sütun için Zipf analizi yap
for column in text_columns:
    print(f"\n=== {column} Sütunu İçin Zipf Analizi ===")
    texts = df[column].dropna().astype(str).tolist()
    if texts:
        perform_zipf_analysis(texts, column)
    else:
        print(f"{column} sütunu boş veya geçersiz.")


# In[ ]:




