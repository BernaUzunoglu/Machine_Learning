######################################################
# Regresyon Modelleri için Hata Değerlendirme
######################################################

##############
# GÖREV
#############
# Çalışanların deneyim yılı ve maaş bilgileri deneyim_maas.xlsx dosyasında verilmiştir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_excel("datasets/deneyim_maas.xlsx")
df.shape
df.info()

# Bağımsız değişken
X = df[["Deneyim_Yılı(x)"]]
# Bağımlı değişken
y = df[["Maaş(y)"]]

#########################################################################################
# ADIM 1 : Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
#########################################################################################
# Bias = 275, Weight= 90 (y’ = b+wx)
# Formül  =>  Maas = 275 + 90 * Deneyim_Yılı(x)
print(" Maas = 275 + 90 * Deneyim_Yılı(x)")
print(f"İlk gözlem için hesaplama  => Maaş = {275+90*5} ")


##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV  # b = sabit, w = ağırlık , x bağımlı değişken

# sabit (b - bias)
b = math.ceil(reg_model.intercept_[0])

# deneyim yılının katsayısı (w1) array döndüğü için [0]
w = round(reg_model.coef_[0][0])

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Maas = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Maas")
g.set_xlabel("Deneyim Yılı")
plt.xlim(0, X["Deneyim_Yılı(x)"].max() + 1)
plt.ylim(bottom=0)
plt.show()

#########################################################################################
# ADIM 2 : Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
#########################################################################################

# Tahmin
y_pred = reg_model.predict(X)
df["Maas_Tahmin(y')"] = reg_model.predict(X)

#########################################################################################
# ADIM 3 : Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız
#########################################################################################

# Tahmin Başarısı İnceleme

df["Hata_(y-y')"] = df["Maaş(y)"] - df["Maas_Tahmin(y')"]
# mean_squared_error(gerçek_değerler, tahmin_edilen_değ)
MSE = mean_squared_error(y, y_pred)
# ortalam hata = 4437.849
y.mean()
y.std()

# RMSE
df["Hata_Kareleri"] = df["Hata_(y-y')"] ** 2
RMSE = np.sqrt(mean_squared_error(y, y_pred))
# 66.617

# MAE
df["Mutlak_Hata(|y-y'|)"] = df["Hata_(y-y')"].abs()
MAE = mean_absolute_error(y, y_pred)
# 54.320

# R-KARE - Veri isetindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesi
# NOT :Değişken sayısı arttıkça  R^2 şişmeye meyillidir.Düzeltilmiş r^2 değerininde göz önünde bulundurulması gerekir.
#     İstatiksel çıktılar ilgilenmiyoruz.(f istatistiği, t istatistiği gibi)
R_Kare = reg_model.score(X, y)
# 0.939

print(f"Modelin MSE(Mean Squard Error) değeri : {MSE} \n"
      f"Modelin RMSE(Root Mean Squard Error) değeri : {RMSE}\n"
      f"Modelin MAE(Mean Absolute Error) değeri : {MAE}\n"
      f"Modelin R-KARE : Veri isetindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesi %: {R_Kare}")