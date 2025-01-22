# =============================================================================
# SCOUTIUM
# =============================================================================
import warnings

# =============================================================================
# İş Problemi
# =============================================================================

# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre,
# yuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

# =============================================================================
# Veri Seti Hikayesi
# =============================================================================

# scoutium_attributes.csv

# task_response_id  : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id          : İlgili maçın id'si
# evaluator_id      : Değerlendiricinin(scout'un) id'si
# player_id         : İlgili oyuncunun id'si
# position_id       : İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#                     1: Kaleci
#                     2: Stoper
#                     3: Sağ bek
#                     4: Sol bek
#                     5: Defansif orta saha
#                     6: Merkez orta saha
#                     7: Sağ kanat
#                     8: Sol kanat
#                     9: Ofansif orta saha
#                     10: Forvet
# analysis_id       : Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id      : Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value   : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

# scoutium_potential_labels.csv

# task_response_id  : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id          : İlgili maçın id'si
# evaluator_id      : Değerlendiricinin(scout'un) id'si
# player_id         : İlgili oyuncunun id'si
# potential_label   : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

# =============================================================================
# Import
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, StandardScaler, RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
# import warnings
#
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)  # Çıktının tek bir satırda olmasını sağlar
pd.set_option("display.float_format", lambda x: "%.5f" %x)
warnings.simplefilter(action='ignore', category=Warning)


# =============================================================================
# Görevler
# =============================================================================

# =============================================================================
# Adım 1 : scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
# =============================================================================

scoutium_attributes = pd.read_csv("datasets/scoutium_attributes.csv", sep=";")
scoutium_potential = pd.read_csv("datasets/scoutium_potential_labels.csv", sep=";")

scoutium_attributes.head()
scoutium_potential.head()

# =============================================================================
# Adım 2 : Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden
# birleştirme işlemini gerçekleştiriniz.)
# =============================================================================

df = pd.merge(scoutium_attributes, scoutium_potential, how="right", on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df.head()
df.shape
# (10730, 9)

# =============================================================================
# Adım 3 : position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
# =============================================================================

df = df.loc[~(df["position_id"] == 1)]
df.shape
# (10030, 9)

# =============================================================================
# Adım 4 : potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
# ( below_average sınıfı tüm verisetinin %1'ini oluşturur)
# =============================================================================

df = df.loc[~(df["potential_label"] == "below_average")]
df.shape
# (9894, 9)

# =============================================================================
# Adım 5 : Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo
# oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
# =============================================================================

# =============================================================================
# Adım 5.1 : İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id”
# ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
# =============================================================================

df.pivot_table("attribute_value", ["player_id", "position_id", "potential_label"], "attribute_id").head()

# =============================================================================
# Adım 5.2 : “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız
# ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
# =============================================================================

df_ = df.pivot_table("attribute_value", ["player_id", "position_id", "potential_label"], "attribute_id").reset_index()
df_["4322"]
df_.head()

df_.columns = [str(col) if not isinstance(col, str) else col for col in df_.columns ]
df_["4322"].head()

# =============================================================================
# Adım 6 : Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini
# (average, highlighted) sayısal olarak ifade ediniz.
# =============================================================================

df_["potential_label"].unique()
df_["potential_label"].value_counts()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df_ = label_encoder(df_,"potential_label")
df_.head()

# =============================================================================
# Adım 7 : Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
# =============================================================================

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degiskenlerin isimlerini verir.
    Not : Kategorik degiskenlerin icerisinde numerik gorunumlu kategorik degiskenler de dahildir.

    Parameters
    ----------
    dataframe : dataframe
        Degisken isimleri alınmak istenen dataframe
    cat_th : int, float
        numerik fakat kategorik olan degiskenler icin sinif esik degeri
    car_th : int, float
        kategorik fakat kardinal degiskenler için sinif esik degeri

    Returns
    -------
    cat_cols : list
        Kategorik degisken listesi
    num_cols : list
        Numerik degisken listesi
    cat_but_car : list
        Kategorik degisken fakat kardinal degisken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam degisken sayisi
    num_but_cat cat_cols'un icerisinde

    """

    # Kategorik olan degiskenlerin secilmesi
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    # Numerik ama aslinda kategorik olan degiskenlerin secilmesi
    num_but_cat = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["Int32", "Int64", "int32", "int64", "float64"] and dataframe[col].nunique() < cat_th]
    # Kategorik degisken ama uniq deger sayisi cok fazla olan degiskenlerin secilmesi
    cat_but_car = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object"] and dataframe[col].nunique() > car_th]
    # Kategorik degiskelerin tamaminin birlestirilmesi
    cat_cols = cat_cols + num_but_cat
    # Uniq degeri fazla olan degerlerin kategorik degiskenlerden cikarilmasi
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Sayisal degiskenlerin secilmesi
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # Sayisal degisken olup kategorik olanlarin sayisal degiskenlerden cikarilmasi
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations :  {dataframe.shape[0]}")
    print(f"Variables :  {dataframe.shape[1]}")
    print(f"cat_cols :  {len(cat_cols)}")
    print(f"num_cols :  {len(num_cols)}")
    print(f"cat_but_car :  {len(cat_but_car)}")
    print(f"num_but_cat :  {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car = grab_col_names(df_)

num_cols = [col for col in num_cols if col not in ["player_id"]]

# =============================================================================
# Adım 8 : Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek
# için StandardScaler uygulayınız.
# =============================================================================

ss = StandardScaler()
df_[num_cols] = ss.fit_transform(df_[num_cols])
df_.head(10)
# =============================================================================
# Adım 9 : Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel
# etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.
# (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
# =============================================================================

# Bagimli degiskenin secilmesi
y = df_["potential_label"]
# Bagimsiz degiskenlerin secilmesi
X = df_.drop(["potential_label","player_id"], axis=1)

def base_models(X, y, scoring = "roc_auc"):
    print("==================== BASE MODEL ====================")
    classifiers = [("LR",LogisticRegression(max_iter=1000)),
                  ("KNN",KNeighborsClassifier()),
                  ("SVC",SVC()),
                  ("CART",DecisionTreeClassifier()),
                  ("RF",RandomForestClassifier()),
                  ("Adaboost",AdaBoostClassifier(algorithm="SAMME")),
                  ("GBM",GradientBoostingClassifier()),
                  ("XGBoost",XGBClassifier(eval_metric="logloss")),
                  ("LightGBM",LGBMClassifier(verbose=-1))
                  # ("CasBoost",CatBoostClassifier(verbose=False))
                  ]

    for name, classifier in classifiers:
        if name == "KNN":
            cv_result = cross_validate(classifier, X, y, cv=3, scoring=scoring)
            print(f"====== {name} ======")
            for score in scoring:
                print(f"{score}:{round(cv_result[f'test_{score}'].mean(), 4)}")
        else:
            cv_result = cross_validate(classifier, X, y, cv=3, scoring=scoring)
            print(f"====== {name} ======")
            for score in scoring:
                print(f"{score}:{round(cv_result[f'test_{score}'].mean(),4)}")

base_models(X, y, ["roc_auc", "f1", "precision", "recall", "accuracy"])

# =============================================================================
# Adım 10 : Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu
# kullanarak özelliklerin sıralamasını çizdiriniz.
# =============================================================================

rf_model = RandomForestClassifier(random_state=17).fit(X,y)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)