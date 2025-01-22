###############################
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
###############################
##############
# İş Problemi
##############
#  Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
#  (average, highlighted) oyuncu olduğunu tahminleme
####################
# Veri Seti Hikayesi
####################
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
# içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

#   scoutium_attributes.csv
#  task_response_id Bir scoutun bir maçta bir takımın kadrosunda ki tüm oyunculara dair değerlendirmelerinin kümesi
#  match_id İlgili maçın id'si
#  evaluator_id Değerlendiricinin(scout'un) id'si
#  player_id İlgili oyuncunun id'si
#  position_id İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#  1: Kaleci
#  2: Stoper
#  3: Sağbek
#  4: Sol bek
#  5: Defansifortasaha
#  6: Merkez ortasaha
#  7: Sağkanat
#  8: Sol kanat
#  9: Ofansifortasaha
#  10: Forvet
#  analysis_id Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
#  attribute_id Oyuncuların değerlendirildiği her bir özelliğin id'si
#  attribute_value Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)


#  scoutium_potential_labels.csv
#  task_response_id Bir scoutun bir maçta bir takımın kadrosunda ki tüm oyunculara dair değerlendirmelerinin kümesi
#  match_id İlgili maçın id'si
#  evaluator_id Değerlendiricinin(scout'un) id'si
#  player_id İlgili oyuncunun id'si
#  potential_label Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedefdeğişken)


import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.evaluate import accuracy_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import advanced_functional_eda as eda
import feature_engineering as fe

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

warnings.simplefilter(action='ignore', category=Warning)

#########################################################
# GÖREVLER
#########################################################
##################################################################################################################
#  Adım 1:  scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
##################################################################################################################
df_attributes = pd.read_csv("datasets/scoutium_attributes.csv", sep=';')
df_potential_labels = pd.read_csv("datasets/scoutium_potential_labels.csv",  sep=';')

##################################################################################################################
#  Adım 2:  Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
#  ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
##################################################################################################################
df = df_attributes.merge(df_potential_labels, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'])
eda.check_df(df)
##################################################################################################################
#  Adım 3:  position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
##################################################################################################################
df.shape
df["position_id"].value_counts()  # 700 tane kaleci var
df = df[df["position_id"] != 1]
##################################################################################################################
#  Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
##################################################################################################################
df["potential_label"].value_counts()  # 136 tane below_average var
df = df.loc[df["potential_label"] != "below_average"]
df.shape
##################################################################################################################
#  Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
#  Adım 5 - 1: İndekste “player_id”,“position_id” ve “potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#  “attribute_value” olacak şekilde pivot table’ı oluşturunuz

#  attribute_id : Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)
##################################################################################################################
df.head()
index = ["player_id", "position_id", "potential_label"]

df_pivot = pd.pivot_table(df, values="attribute_value", index=index, columns="attribute_id")
df_pivot.head(10)

##################################################################################################################
#  Adım 5 - 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
##################################################################################################################
df_pivot = df_pivot.reset_index()
df_pivot.info()
# attribute_id sütunlarının isimlerini stringe çevirelim
type(df_pivot.columns[5])
df_pivot.columns = df_pivot.columns.map(str)
##################################################################################################################
#  Adım 6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average(0) = 215, highlighted(1) = 56) sayısal olarak ifade ediniz.
##################################################################################################################
df_pivot["potential_label"].value_counts()
df_pivot = fe.label_encoder(df_pivot, "potential_label")

##################################################################################################################
#  Adım 7:  Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
##################################################################################################################
num_cols = [col for col in df_pivot.columns if col not in ['player_id', 'position_id', 'potential_label']]

##################################################################################################################
#  Adım 8:  Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
##################################################################################################################
df_pivot = fe.num_cols_standardization(df_pivot, num_cols, "ss")
df_pivot.head(10)
##################################################################################################################
#  Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
#  geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
##################################################################################################################

y = df_pivot["potential_label"]
X = df_pivot.drop(["potential_label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier(max_features="sqrt")),
          ('GBM', GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier(objective='reg:squarederror'))]

for name, classifier in models:
    cv_results = cross_validate(classifier, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    print(f"##############    {name}    ##############")
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_f1'].mean()} ")
    print(f"F1: {cv_results['test_accuracy'].mean()}")
    print(f"Roc_AUC: {cv_results['test_roc_auc'].mean()}")

# ##############    RF    ##############
# Accuracy: 0.8672053872053871
# Precision: 0.8316666666666667
# Recall: 0.6045518207282913
# F1: 0.8672053872053871
# Roc_AUC: 0.9109584214235376


# max_features :özelliklerin karekökü kadar rastgele seçilmesini sağlar.
rf_model = RandomForestClassifier(max_features="sqrt", random_state=17).fit(X, y)
rf_model.get_params()

rf_params = {'n_estimators': [50, 100, 200],
             'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}


# GridSearchCV kullanarak en iyi hiperparametreleri bulma
grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi parametreleri ve en iyi skorları yazdırma
# En iyi parametreler: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
print("En iyi parametreler:", grid_search.best_params_)
# En iyi skor: 0.9121564482029598
print("En iyi skor:", grid_search.best_score_)

# En iyi modeli seçme ve test seti üzerinde değerlendirme
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test seti doğruluğu: {accuracy}")

random = X.sample(1, random_state=45)

best_rf.predict(random)

##################################################################################################################
#  Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz
##################################################################################################################

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
plot_importance(best_rf, X)



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


#  Precision 0'a bölünme hatası çözme
# Her sınıf için precision değerlerini hesapla
precision_values = precision_score(y, y_pred, average=None, zero_division=0)
print(f'Precision Values: {precision_values}')

# Precision değeri 0 olan veya hataya neden olan sınıfları belirle
problematic_classes = np.where(precision_values == 0)[0]
print(f'Problematic Classes: {problematic_classes}')