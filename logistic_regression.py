######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis - Keşifçi veri analizi
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerleri hesaplanan eşik değeri ile değiştirme fonk.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


######################################################
# Exploratory Data Analysis - EDA
######################################################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
##########################
# Target'ın Analizi
##########################
# Bağımlı değişken
df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

# Bütün veri setindeki yüzdelik oranı
100 * df["Outcome"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################

df.head()
df.describe().T

# Kan basıncı değeri için
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

# Numerik kolonları görselleştirme fonksiyonu
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

# Bağımlı değişkeni çıkartalım
cols = [col for col in df.columns if "Outcome" not in col]


for col in cols:
    plot_numerical_col(df, col)

df.describe().T

##########################
# Target vs Features
##########################

# Diyabet olanların hamilelik ortalamasına bakalım
df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)


######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()

# Eksik değerlere bakalım
df.isnull().sum()

df.describe().T

# Aykırı değerlere bakalım
for col in cols:
    print(col, check_outlier(df, col))

# Sadece Insulin değerinde aykırılık var . Aykırı değerleri eşik değerleri ile değiştirelim.
replace_with_thresholds(df, "Insulin")

# Ölçeklendirme - RobustScaler aykırı değerlerden etkilenmez.
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


######################################################
# Model & Prediction
######################################################

y = df["Outcome"]  #Bağımlı değişken

X = df.drop(["Outcome"], axis=1)  #Bağımsız değişken

log_model = LogisticRegression().fit(X, y)

# bias - sabit değeri
log_model.intercept_
# w ağırlık katsayı değerleri
log_model.coef_

y_pred = log_model.predict(X)
# Tahmin edilen 10 değer
y_pred[0:10]
# Gerçek değerlere bakalım
y[0:10]

######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))


# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939
# NOT : Model bütün veride kuruldu ve tahmin edildi.

######################################################
# Model Validation: Holdout (Nasıl Çalışır ? Veri setini iki parçaya böl biriyle modeli eğit diğeri ile test et )
# (Çünkü kurulan model bütün veride kuruldu ve tahmin yapıldı doğrulanması gerek.)
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

# Modeli kur
log_model = LogisticRegression().fit(X_train, y_train)
#  Kurulan modelde tahminde bulunalım.
y_pred = log_model.predict(X_test)
# 1 sınıfına ait olma olasılıklarına bakalım
y_prob = log_model.predict_proba(X_test)[:, 1]

# y_test test veri setindeki gerçek değerler ile tahmin edilen y değerleri
print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# ROC eğrisini oluşturalım
RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

#  Ufsk tefek değişen değerlerde random olarak veriden işlem yapıldığını unutmayalım.

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################
# NOT: Model doğrularken hangi train,test yani hangi 80 ne 20 problemi yaşıyorduk.
# Bu doğrulamam sürecini en doğru şekilde ele almak için bu işlemi gerçekleştireceğiz.
# Houldout yöntetmi 10 katlı şekilde yapmak demektir.

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327
# Kayda değer bir model yorumu yapılabilir.

######################################################
# Prediction for A New Observation
######################################################
#Diyabet olup olamayacağını tahmin edelim.
X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)
# ÇIKTI : Out[86]: array([1] bu seçilen kişi diyabettir tahmini yapılmıştır.
















