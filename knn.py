################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation  - Bütün veri setinde tahmin işlemleri ve doğruluğu
################################################

# Confusion matrix için y_pred: bütün gözlem değerleri için tahminler
y_pred = knn_model.predict(X)

# AUC için y_prob: ROC eğrisinin altında kalan alan
#  1 olması olasılık değeri
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# precisiom : 1 olarak tahmin ettiklerimizin başarısı - 1 sınıfına yönelik tahmin başarısı
# recall : gerçekte 1 olanları 1 tahmin etme olsaılığı
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob)
# 0.90

# Aynı veri setinde eğitilip tahmin edildiği için Holdout yada çapraz doğrulama ile modeli doğrulamamız gerekir.
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# 0.73
# 0.59
# 0.78

# Başarı skorları nasaıl artırılabilir ?
# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme(İşlemler detayladırılabilir.)
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

# En optimum  komşuluk sayısını bulmak.
knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

# Optimum komşuluk sayısını bulduk.
knn_gs_best.best_params_

################################################
# 6. Final Model
################################################
# Yukarda bulunan optimum komşuluk sayısına göre tekrar model kurmak gerek.
# ** key-value çiftlerini argüman olarak kullanmak için
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)











