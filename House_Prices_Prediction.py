###############################
# House Prices Prediction
###############################
##############
# İş Problemi
##############
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak, farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

####################
# Veri Seti Hikayesi
####################
# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması
# da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait
# olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation
# değerleri sizin tahmin etmeniz beklenmektedir.

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import advanced_functional_eda as eda
import feature_engineering as fe

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

warnings.simplefilter(action='ignore', category=Warning)

variable_explanations = {
    "SalePrice": "Mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalıştığınız hedef değişkendir.",
    "MSSubClass": "Bina sınıfı",
    "MSZoning": "Genel imar sınıflandırması",
    "LotFrontage": "Mülke bağlı olan yolun lineer feet cinsinden uzunluğu",
    "LotArea": "Arsanın metrekare cinsinden büyüklüğü",
    "Street": "Yol erişim türü",
    "Alley": "Arka yol erişim türü - Grvl: Çakıl ,toprak ve Pave : Döşeme,kaplama",
    "LotShape": "Mülkün genel şekli",
    "LandContour": "Mülkün düzlüğü",
    "Utilities": "Mevcut altyapı hizmetleri",
    "LotConfig": "Arsa konfigürasyonu",
    "LandSlope": "Mülkün eğimi",
    "Neighborhood": "Ames şehir sınırları içindeki fiziksel konumlar",
    "Condition1": "Ana yol veya demiryoluna yakınlık",
    "Condition2": "Ana yol veya demiryoluna yakınlık (ikinci bir yakınlık varsa)",
    "BldgType": "Konut tipi",
    "HouseStyle": "Konut tarzı",
    "OverallQual": "Genel malzeme ve işçilik kalitesi",
    "OverallCond": "Genel durum değerlendirmesi",
    "YearBuilt": "İlk inşaat tarihi",
    "YearRemodAdd": "Yenileme tarihi",
    "RoofStyle": "Çatı tipi",
    "RoofMatl": "Çatı malzemesi",
    "Exterior1st": "Evin dış kaplaması",
    "Exterior2nd": "Evin dış kaplaması (birden fazla malzeme varsa)",
    "MasVnrType": "Tuğla kaplama türü",
    "MasVnrArea": "Tuğla kaplama alanı (metrekare cinsinden)",
    "ExterQual": "Dış malzeme kalitesi",
    "ExterCond": "Dış malzemenin mevcut durumu",
    "Foundation": "Temel tipi",
    "BsmtQual": "Bodrum yüksekliği",
    "BsmtCond": "Bodrumun genel durumu",
    "BsmtExposure": "Bahçeye açılan veya yürüme seviyesinde olan bodrum duvarları",
    "BsmtFinType1": "Bodrumun birinci bitirilmiş alanının kalitesi",
    "BsmtFinSF1": "Birinci bitirilmiş alanın metrekare cinsinden büyüklüğü",
    "BsmtFinType2": "İkinci bitirilmiş alanın kalitesi (varsa)",
    "BsmtFinSF2": "İkinci bitirilmiş alanın metrekare cinsinden büyüklüğü",
    "BsmtUnfSF": "Bitirilmemiş bodrum alanı (metrekare cinsinden)",
    "TotalBsmtSF": "Toplam bodrum alanı (metrekare cinsinden)",
    "Heating": "Isıtma türü",
    "HeatingQC": "Isıtma kalitesi ve durumu",
    "CentralAir": "Merkezi klima",
    "Electrical": "Elektrik sistemi",
    "1stFlrSF": "Birinci kat metrekare cinsinden büyüklüğü",
    "2ndFlrSF": "İkinci kat metrekare cinsinden büyüklüğü",
    "LowQualFinSF": "Düşük kaliteli bitirilmiş alan (tüm katlar)",
    "GrLivArea": "Yer seviyesinin üstündeki (zemin üstü) yaşam alanı metrekare cinsinden büyüklüğü",
    "BsmtFullBath": "Bodrum katındaki tam banyolar",
    "BsmtHalfBath": "Bodrum katındaki yarım banyolar",
    "FullBath": "Zemin üstündeki tam banyolar",
    "HalfBath": "Zemin üstündeki yarım banyolar",
    "Bedroom": "Bodrum seviyesi üstündeki yatak odası sayısı",
    "Kitchen": "Mutfak sayısı",
    "KitchenQual": "Mutfak kalitesi",
    "TotRmsAbvGrd": "Zemin üstündeki toplam oda sayısı (banyolar hariç)",
    "Functional": "Ev işlevsellik değerlendirmesi",
    "Fireplaces": "Şömine sayısı",
    "FireplaceQu": "Şömine kalitesi",
    "GarageType": "Garaj konumu",
    "GarageYrBlt": "Garaj inşa yılı",
    "GarageFinish": "Garajın iç bitirme durumu",
    "GarageCars": "Garaj kapasitesi (araba sayısı)",
    "GarageArea": "Garaj alanı (metrekare cinsinden)",
    "GarageQual": "Garaj kalitesi",
    "GarageCond": "Garaj durumu - Ex (Excellent): Mükemmel kalite ,Gd (Good): İyi kalite ,Fa (Fair): Orta kalite TA (Typical/Average) PO(Poor): Kötü) ",
    "PavedDrive": "Asfalt kaplı sürüş yolu",
    "WoodDeckSF": "Ahşap güverte alanı (metrekare cinsinden)",
    "OpenPorchSF": "Açık veranda alanı (metrekare cinsinden)",
    "EnclosedPorch": "Kapalı veranda alanı (metrekare cinsinden)",
    "3SsnPorch": "Üç mevsimlik veranda alanı (metrekare cinsinden)",
    "ScreenPorch": "Ekranlı veranda alanı (metrekare cinsinden)",
    "PoolArea": "Havuz alanı (metrekare cinsinden)",
    "PoolQC": "Havuz kalitesi - Ex (Excellent): Mükemmel kalite ,Gd (Good): İyi kalite ,Fa (Fair): Orta kalite",
    "Fence": "Çit kalitesi",
    "MiscFeature": "Diğer kategorilere dahil olmayan çeşitli özellikler",
    "MiscVal": "Çeşitli özelliklerin $ değeri",
    "MoSold": "Satış ayı",
    "YrSold": "Satış yılı",
    "SaleType": "Satış türü",
    "SaleCondition": "Satış durumu"
}

#########################################################
# Görev 1 : Keşifçi Veri Analizi - EDA
# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
#########################################################
df_train = pd.read_csv("datasets/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("datasets/house-prices-advanced-regression-techniques/test.csv")
df = pd.concat([df_train, df_test])
df.head(25)
eda.check_df(df)
############################################################################
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
############################################################################
# ID colonunu çıkaralım
df.drop(columns=['Id'], inplace=True)
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# GarageCars değişkeni araba sayısı temsil etmektedir. Sayısal bir değişken kategorik olanlara alındı. çıkaralım ve num_cols ekleyelim.
cat_cols = [col for col in cat_cols if col not in ["GarageCars", "Fireplaces", "PoolArea"]]
num_cols.extend(["GarageCars", "Fireplaces", "PoolArea"])
# num_cols.append("GarageCars")


#############################################################################
# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
#############################################################################

# Bağımlı değişkeni çıkaralım
num_cols = [col for col in num_cols if "SalePrice" not in col]

#####################################################################################
# Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
#####################################################################################
for col in cat_cols:
    eda.cat_summary(df, col)

for col in num_cols:
    eda.num_summary(df, col)

remove_columns_list = ["Street", "Alley", "LandSlope", "LandContour", "Utilities",  "MiscFeature", "Neighborhood", "Fence", "Heating", "PoolQC", "PoolArea"]


cat_cols = [col for col in cat_cols if col not in ["Street", "Alley", "LandSlope", "LandContour", "Utilities",  "MiscFeature", "Fence", "Heating", "PoolQC"]]
len(cat_cols)
num_cols = [col for col in num_cols if "PoolArea" not in col]
len(num_cols)
##########################################################################
# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
##########################################################################
for col in cat_cols:
    eda.detail_target_summary_with_cat(df, "SalePrice", col)

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
############################################
# Adım 6: Aykırı gözlem var mı inceleyiniz.
############################################
for col in num_cols:
    if fe.check_outlier(df, col):
        print(col)
###########################################
# Adım 7: Eksik gözlem var mı inceleyiniz.
###########################################
na_columns, na_columns_detail = fe.missing_values_table(df, na_name=True, na_cols_detail=True)

#########################################################
# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
###################################################################

# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    if fe.check_outlier(df, col):
        fe.replace_with_thresholds(df, col)

##################################
# Eksik Değer İşlemleri
###############################
# Null değerlerin analizi yaparken sütuna özel analiz yapmamız gerekiyor. Çünkü örneğin evin havuzu yok, bu nedenle PoolQC için null değer

na_columns
len(na_columns)
# Bağımlı değişken na_columns listesinden çıkaralım.
na_columns = [col for col in na_columns if "SalePrice" not in col]

# Kategorik ve Numereik boş değişkenleri bulalım
na_num_cols = []
na_cat_cols = []
for col in na_columns:
    na_num_cols.append(col) if col in num_cols else na_cat_cols.append(col)

# numerik NaN olan kategorilerde benzersiz değerlere bakmak mantıklı olmayacaktır. O yüzden kategorilerin benzersiz değerlerine bir bakalım.
for col in na_cat_cols:
    print(f"{col}: {variable_explanations[col]}") if col in variable_explanations else None
    print(pd.DataFrame({col: df[col].unique()}))
    print("#################################")


# GarageType sütununda NaN olan satırları seçin ve diğer ilgili sütunları da görüntüleyin
df_nan_garage = df[df['GarageType'].isnull()][['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']].head(20)
# Garj alanı(metre kare) değeri 0 olması garaç olmadığı anlamına gelmektedir. Garaj ile ilgili kategoriklerde NaN değerler NG : No Garage ile dolduralım.
df["GarageType"].fillna("NG", inplace=True)
df["GarageFinish"].fillna("NG", inplace=True)
df["GarageQual"].fillna("NG", inplace=True)
df["GarageCond"].fillna("NG", inplace=True)

# Şömine olmayan değerlere ulaşalım
df_nan_fireplaces = df[df["Fireplaces"] == 0]["FireplaceQu"].head(30)
len(df_nan_fireplaces)  # 1420 adet gözlem var. FireplaceQu değişkeninin Nan değerlerini NFP: No Fire PLaces ile dolduralım
df["FireplaceQu"].fillna("NFP", inplace=True)

# Electrical değiikeninde bir değişken Nan değerinde gözlemi silebiliriz.
index=df[df["Electrical"].isnull()==True].index
df.drop(index, inplace=True)


# Basement(Bodrum) ile alakalı değişkenler
df_nan_basement = df[df["TotalBsmtSF"] == 0]
len(df_nan_basement)  # çıkan gözlem değeri diğer bodrum değişkenleri ile aynı. Bodrum yok NB  ile doldurslım.

df["BsmtQual"].fillna("NB", inplace=True)
df["BsmtCond"].fillna("NB", inplace=True)
df["BsmtExposure"].fillna("NB", inplace=True)
df["BsmtFinType1"].fillna("NB", inplace=True)
df["BsmtFinType2"].fillna("NB", inplace=True)

# Tuğla kaplama türü değişkenin de 1766 tane eksik değer var. Df den çıkaramayız çok veri kaybı olur.  HSMV = hasn`t masonrt veneer il dolduralım.
#  MasVnrArea modelde önemli değişkenlerde çıkıyor.
df["MasVnrType"].fillna("HSMV", inplace=True)

# Numerik Nan Değerlerin doldurulması
# Alley arka yol erişim türü demek LotFrontage mülke bağlı olan yol uzunluğu demek burdan yola çıkarsak Alley olmayanların yol uzunluğu da olmaz diyerek LotFrontage 0 ile dolduralım.
df["LotFrontage"].fillna(0, inplace=True)
df["GarageYrBlt"].fillna(0, inplace=True)  # sayısal değerli
df["GarageCars"].fillna(0.0000, inplace=True)
df["MasVnrArea"].fillna(0, inplace=True)

# Boş kalan diğer değerleri analiz etmeden önce dataframeden çıkarılması gereken değişkenleri çıkaralım.
df = df.drop(columns=remove_columns_list)

# Geriye 1-2 adet boş değerli gözlemlermiz kalıyor onları dataframemimizden çıkaralım.(21 adet boş gözlem değeri kaldı.)
missing_values = [col for col in df.columns if df[col].isnull().sum() > 0 and col != 'SalePrice']
for col in missing_values:
    index = df[df[col].isnull() == True].index
    df.drop(index, inplace=True)

########################################
# Adım 2: Rare Encoder uygulayınız.
########################################

# Kategorik kolonların dağılımının inceleyelim
fe.rare_analyser(df, "SalePrice", cat_cols)

# Nadir sınıfların tespitini yapalım
df = fe.rare_encoder(df, 0.01)
df["SaleType"].value_counts()
df["KitchenAbvGr"].value_counts()  # kitchen rare düşmedi çünkü rareiçin verdiğimiz yüzdelik thresholdunda küçük bir oranan sahip değil.

########################################
# Adım 3: Yeni değişkenler oluşturunuz.
########################################
# Toplam Banyo sayısı
df["TotalBaths"] = df["BsmtFullBath"] + (df["BsmtHalfBath"] * 0.5) + df["FullBath"] + (df["HalfBath"] * 0.5)

# Garaj varsa 1 yoksa 0
df['Isgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

# Şömine varsa 1 Yoksa 0
df['Isfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Havuz varsa 1 yoksa 0
# df['Ispool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

# Ev iki katlı ise 1 yoksa 0
df['Issecondfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

# Açık veranda varsa 1 yoksa 0
df['IsOpenPorchSF'] = df['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)

# Kapalı veranda varsa 1 yoksa 0
df['IsEnclosedPorch'] = df['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)

df["BuildAge"] = df["YrSold"] - df["YearBuilt"]

#################################################
# Adım 4: Encoding işlemlerini gerçekleştiriniz.
#################################################

df = fe.one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

#################################################
# Görev 3 : Model Kurma
# Adım 1:  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
##################################################################################################
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice']
X = train_df.drop("SalePrice", axis=1)

#############################################################################
# Adım 2:  Train verisi ile model kurup, model başarısını değerlendiriniz.
#############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          # ('GBM', GradientBoostingRegressor()),
          # ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# Değişkenlerin çıkarılmadan önceki model değerleri
# RMSE: 51184.0764 (LR)
# RMSE: 44654.2413 (KNN)
# RMSE: 41086.6008 (CART)
# RMSE: 29979.7223 (RF)
# RMSE: 30746.7908 (XGBoost)
# RMSE: 28526.4258 (LightGBM)

# Şimdiki model değerleri
# RMSE: 50053.3982 (LR)
# RMSE: 44564.0392 (KNN)
# RMSE: 43806.4909 (CART)
# RMSE: 30498.6913 (RF)
# RMSE: 30234.2205 (XGBoost)
# RMSE: 28374.2984 (LightGBM)

df_train['SalePrice'].mean()  #180921.19589041095
df_train['SalePrice'].std()  # 79442.50288288663

###################################################################################################################
# Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'untersini(inverse) almayıunutmayınız.
#####################################################################################################################

# Hedef değişkenin logaritmik dönüşümü
y = np.log1p(train_df['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

# Tahminleri yapma ve geri dönüştürme
# Yapılan LOG dönüşümünün tersinin (inverse'nin) alalım
inverse_y = np.expm1(y_pred)
inverse_y_test = np.expm1(y_test)

# Modelin performansını değerlendirme
rmse = np.sqrt(mean_squared_error(inverse_y_test, inverse_y))
print(f'RMSE: {rmse}')

######################################################
# Adım3: Hiper paremetre optimizasyonu gerçekleştiriniz.
######################################################

gbm_model = LGBMRegressor(random_state=17)
gbm_model.get_params()


gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=5, scoring="neg_mean_squared_error")))


#################################################
# Adım4: Değişken önem düzeyini inceleyeniz.
#################################################
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

plot_importance(gbm_final, X, 30)
plot_importance(lgbm, X, 30)


##################################################################################################
# Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturup sonucunuzu yükleyiniz.
##################################################################################################
model = LGBMRegressor(random_state=17)
model.fit(X, y)
test_df["SalePrice"]
# Test setindeki boş SalePrice değerlerini tahmin etme
X_test = test_df.drop("SalePrice", axis=1)
test_pred_log = model.predict(X_test)


# Tahminleri Kaggle formatına uygun DataFrame'e dönüştürme
predict_df = pd.DataFrame({
    "Id": test_df.index,
    "SalePrice": test_pred_log
})

# CSV dosyası olarak kaydetme
predict_df.to_csv('submission.csv', index=False)