###############################
# Telco Churn Prediction
###############################
##############
# İş Problemi
##############
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
############################
# Veri Seti Hikayesi
############################
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını ve ya hizmete kaydolduğunu gösterir.

# CustomerId : Müşteri id'si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, İnternet hizmeti yok)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import advanced_functional_eda as eda
import feature_engineering as fe


from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/Telco-Customer-Churn.csv")

#########################################################
# Görev 1 : Keşifçi Veri Analizi - EDA
#########################################################
# Genel Resim
eda.check_df(df)
#########################################################
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
#########################################################
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)
# Bağımlı değişkeni çıkaralım
cat_cols = [col for col in cat_cols if "Churn" not in col]


############################################################################
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
############################################################################
# TotalCharges (Müşteriden tahsil edilen toplam tutar ) - object tipinde float olmalı
# Değişken verilerinde boşluk karakterleri var onları NaN değerlerine dönüştürelim.
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype('float64')

#####################################################################################
# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
#####################################################################################
for col in cat_cols:
    eda.cat_summary(df, col)

for col in num_cols:
    eda.num_summary(df, col)

#############################################################################
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
#############################################################################

for col in cat_cols:
    eda.target_summary_with_cat_crosstab(df, "Churn", col)

#############################################
# Adım 5: Aykırı gözlem var mı inceleyiniz.
#############################################

for col in num_cols:
    print(col, fe.check_outlier(df, col))
#############################################
# Adım 6: Eksik gözlem var mı inceleyiniz.
#############################################
df.isnull().values.any()  # TotalCharges - 11 tane eksik değer var.
df.isnull().sum()
fe.missing_values_table(df)
msno.bar(df)

##################################
# Görev 2 : Feature Engineering
#################################

####################################################################
# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# Eksik değerler için eksik verileri çıkarma - basit atama yöntemleri - tahmine dayalı atama yapılabilir
####################################################################
df.shape
df.isnull().sum()
#  7043 gözlemde sadece 11 gözlem eksik değere sahip veri setini etkilemeyeceğinden direk çıkarabiliriz.
df = df[df.notnull().all(axis=1)]

########################################
# Adım 2: Yeni değişkenler oluşturunuz.
########################################
df.head(15)

# Müşterinin kaç yıldır hizmeti kullandığını bir sütunda yıl bazında gösterelim.
df['YearsAsCustomer'] = (df['tenure'] // 12).astype(str) + ' yıl ' + (df['tenure'] % 12).astype(str) + ' ay'

# Müşterilerin ödedikleri aylık harcamanın toplam harcamalarına oranını
df['ChargeRatio'] = df['MonthlyCharges'] / df['TotalCharges']

# Stream TV yayını olan müşterilerin Stream Movie hizmeti alıp almadığını incelemek için
df['Strem_TV_Movie'] = df[['StreamingTV', 'StreamingMovies']].apply(lambda row: '-'.join(row.values), axis=1)

#  Tenure ve contract değişkenleri üzerinden müşteri sadakat sınıflandırması yapabiliriz.
#  Örneğin, 24 aydan uzun süre abone olanları "Yüksek Sadakat", diğerlerini "Düşük Sadakat" olarak tanımlayabiliriz
df['CustomerLoyalty'] = df.apply(lambda row: 'Yüksek Sadakat' if (row['tenure'] >= 24 and row['Contract'] != 'Month-to-month') else 'Düşük Sadakat', axis=1)


##################################################
# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
##################################################
binary_cols = eda.get_binary_columns(df, cat_cols)
non_binary_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols = binary_cols + non_binary_cols

dff = fe.one_hot_encoder(df, ohe_cols, drop_first=True)
dff.head()

############################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
#############################################################
dff = fe.num_cols_standardization(dff, num_cols, "rs")
dff.head()
############################
# Görev 3 : Modelleme
# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
####################################################################################################################


y = dff["Churn"]
dff["Churn"] = fe.label_encoder(dff, "Churn")
y = fe.label_encoder(new_df, "Churn")
X = dff.drop(["Churn", "customerID"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)
dff.head()
knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)
def model_evaluation(model, X, y, cv=10):
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {model} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


model_evaluation(knn_model,X,y)
cv_results = cross_validate(knn_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
print(f"########## {knn_model} ##########")
print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")

models = [("LR", LogisticRegression(random_state=42)),
          ("KNN", KNeighborsClassifier()),
          ("CART", DecisionTreeClassifier(random_state=42)),
          ("RF", RandomForestClassifier(random_state=42)),
          ("SVM", SVC(gamma="auto", random_state=42)),
          ("XGB", XGBClassifier(random_state=42)),
          ("LightGBM", LGBMClassifier(random_state=42, force_row_wise=True)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=42))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


##############################################################################
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin
# ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.
##############################################################################