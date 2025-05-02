
# Yapay-zeka-devi-k-t-phane-projesi-
Kütüphane Projesi: Kitap Konusu ile Bölüm Eşleşmesi Problem: Kitabın bölümlerini genel konusuna göre ne kadar ilgili olduğunu ölçmek. Veri Kaynağı: Project Gutenberg tam metin kitaplar Adımlar: 1. Kitabın genel açıklaması ile her bölüm ayrı ayrı vektörleştirilir. 2. Cosine similarity hesaplanır. 3. Tutarsız bölümler işaretlenir.
Amaç:

Bu projenin amacı, bir kitabın genel açıklaması ile o kitaba ait bireysel bölümlerin içeriklerinin ne kadar uyumlu olduğunu otomatik olarak ölçmektir. Böylece, kitap içerisindeki tutarsız veya konudan sapmış bölümler tespit edilebilir.
Veri Kaynağı:

Açık erişimli dijital kitap arşivlerinden biri olan Project Gutenberg üzerinden elde edilen tam metin kitaplar kullanılmıştır. Örnek olarak Lewis Carroll’un Alice’s Adventures in Wonderland adlı eseri seçilmiştir.
Yöntem ve Adımlar:

1. Veri Hazırlığı:

Kitabın genel açıklaması (konusu) belirlenmiştir.

Kitabın ilk 5 bölümü örnek olarak alınmış ve her biri ayrı bir metin olarak düzenlenmiştir.


2. Veri Ön İşleme (Text Preprocessing):

Metinlerin sayısal işlemlere uygun hale getirilmesi için aşağıdaki adımlar uygulanmıştır:

Küçük harfe dönüştürme,

Noktalama işaretlerinin ve stopword’lerin (gereksiz kelimeler) temizlenmesi,

Lemmatizasyon işlemi ile kelimelerin köklerine indirgenmesi.
3. Vektörleştirme (TF-IDF):

Kitabın genel açıklaması ve her bölüm, TF-IDF (Term Frequency–Inverse Document Frequency) yöntemi ile vektörlere dönüştürülmüştür.

TF-IDF, kelimelerin belgeler arasındaki önemini belirlemeye yarayan bir istatistiksel yöntemdir.


4. Cosine Similarity ile Benzerlik Ölçümü:

Kitap açıklaması ile her bölüm arasındaki cosine similarity (kosinüs benzerliği) değeri hesaplanmıştır.

Bu sayede her bölümün kitap konusuyla ne kadar ilişkili olduğu sayısal olarak ölçülmüştür.


5. Sonuçların Yorumlanması:

Eşik değer (örneğin 0.15) altında kalan benzerlik skorlarına sahip bölümler “Tutarsız” olarak işaretlenmiştir.

Sonuçlar bir tablo halinde sunulmuştur.
Kullanılan Teknolojiler:

Programlama Dili: Python

Kütüphaneler: NLTK, scikit-learn, pandas

Ortam: Jupyter Notebook
Sonuç:

Bu proje sayesinde, büyük metinlerin bölümlerinin içerikle ne kadar uyumlu olduğu otomatik olarak değerlendirilebilmektedir. Bu yaklaşım, kitap redaksiyonu, özetleme, içerik kontrolü gibi birçok alanda kullanılabilir hale getirilebilir.

