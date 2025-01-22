###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )  müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem, 13 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type : 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.


import pandas as pd
import datetime as dt
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering


import advanced_functional_eda as eda
import feature_engineering as fe
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
warnings.simplefilter(action='ignore')
###############################################################
# GÖREV 1 . Veriyi Hazırlama
###############################################################

###############################################################
#  Adım1:  flo_data_20K.csv verisini okutunuz.
###############################################################
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

eda.check_df(df)

# Tarih değişkenlerinin tipini düzeltelim.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
###############################################################
#  Adım2:  Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
###############################################################

analysis_date = df["last_order_date"].max() + dt.timedelta(days=2)

df["recency"] = (analysis_date - df["last_order_date"]).dt.days  # en son kaç gün önce alışveriş yaptı
#  II . YOL
# df['recency1'] = df['last_order_date'].apply(lambda x: (analysis_date - x).days)
# Müşterinin e-ticaret sitesi/mağaza ile ilk kontağından bu yana geçen zaman olarak geçer.

# tenure hesaplama
def calculate_tenure(row):
    delta = relativedelta(analysis_date, row["first_order_date"])
    years = delta.years
    months = delta.months
    return years, months


df["tenure_days"] = (analysis_date - df["first_order_date"]).dt.days
df["tenure"], df["tenure_months"] = zip(*df.apply(calculate_tenure, axis=1))

# Müşterinin ne sıklıkla alışveriş yaptığını, ne sıklıkla siteye giriş yaptığını gösteren metriktir.
df['frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

# Müşterinin harcamalarının toplamıdır. E-ticaret sitesine getirdiği ciro, aldığı hizmetler sonrası toplanan getiri olarak da tanımlanır.

df['monetary'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

# kümeleme algoritmaları çoğunlukla sayısal verilerle ilgilenmekte olduğu için sadece int64 ve float64 tipindeki sütunları seçelim
new_df = df.select_dtypes(include=['int64', 'float64'])
new_df.head()
new_df.info()
new_df.shape

# Numerik kolonlara bir bakalım
new_df[new_df.columns].describe().T

###############################################################
# GÖREV 2 .  K-Means ile Müşteri Segmentasyonu
###############################################################

###############################################################
# Adım 1: Değişkenleri standartlaştırınız.
###############################################################
# Numerik kolonları standartlaştıralım. - MinMaxScaler ile
new_df = fe.num_cols_standardization(new_df, new_df.columns, "mms")
new_df.head()


###############################################################
#  Adım 2: Optimum küme sayısını belirleyiniz.
###############################################################
kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(new_df)
    ssd.append(kmeans.inertia_)

kmeans.get_params()
# Küme sayısı
kmeans.n_clusters
# Kümelerin merkezleri
kmeans.cluster_centers_
# Küme etiketleri
kmeans.labels_
# SSE - örneklerin en yakın küme merkezine olan uzaklıklarının karelerinin toplamını ifade eder.
kmeans.inertia_

# Dirsek Yöntemi - Elbow:
#  Kümelere nasıl ihtiyaç duyacağımız konusunda kafamız karıştığında kullanışlı olur. Grafiğimiz dirseğe benziyor ve bu dirsek noktasını belirlemekte bize yardımcı olur.
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
# Elbow - dirsek yöntemi : Kümeleme için optimum noktayı verir.
elbow = KElbowVisualizer(kmeans, k=(2, 15))
elbow.fit(new_df)
elbow.show()

elbow.elbow_value_

###############################################################
# Adım 3:  Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
###############################################################
# Elbow değeri ile modeli fit ediyoruz.
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(new_df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

# Kümeleri bir değişken de tutalım
clusters_kmeans = kmeans.labels_
# Dataframe de küme değerleri yazacağımız bir sütun oluşturalım.
df["cluster"] = clusters_kmeans

#  0 ifadesinden kurtulmak için 1 değerinin ekleyelim.
df["cluster"] = df["cluster"] + 1

df[df["cluster"] == 5].head(10)

###############################################################
#  Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
###############################################################
eda.ratio_column(df, "cluster")

df.groupby("cluster").agg(["count", "mean", "median"]).T

###############################################################
# GÖREV 3.  Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################


###############################################################
#  Adım 1: Görev2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
###############################################################
# linkage birleştirici bi clustering yöntemi
hc_average = linkage(new_df, "average")

plt.figure(figsize=(17, 8))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.7, color='b', linestyle='--')
plt.axhline(y=1.5, color='b', linestyle='--')
plt.show(block=True)

###############################################################
#  Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
###############################################################
# Birleştirici kümeleme yöntemini kullanalım

cluster = AgglomerativeClustering(n_clusters=8, linkage="average")

clusters = cluster.fit_predict(new_df)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

###############################################################
#  Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
###############################################################
eda.ratio_column(df, "hi_cluster_no")

df.groupby("hi_cluster_no").agg(["count", "mean", "median"]).T