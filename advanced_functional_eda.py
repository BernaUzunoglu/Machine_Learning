import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Genel Resim
def check_df(dataframe, head=5):
    print("############### Shape  ###############")
    print(dataframe.shape)
    print("############### Types  ###############")
    print(dataframe.dtypes)
    print("############### Head  ###############")
    print(dataframe.head(head))
    print("############### Tail  ###############")
    print(dataframe.tail(head))
    print("############### NA  ###############")
    print(dataframe.isnull().sum())
    print("############### Quantiles  ###############")
    print(dataframe.describe([0, 0.85, 0.50, 0.95, 0.99, 1]).T)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki, kategorik, numeric ve kategorik fakat kardinal değişkenlerin isimlerini verir
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th : int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int,float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numeric değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + car_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    # int ve float değişkenlerde eşssiz class sayısını(nunique) bul
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat

    #  Kategorik gibi duran fakat kardenel olan değişkenlervarsa kategorik kolonlardan çıkarılması gerek
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Veri setinden numeric değerleri nasıl seçeriz?
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]

    # num_cols da olup cat_cols da olmayanları seçelim
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def get_binary_columns(dataframe, cat_cols):
    #   İki sınıflı kategorik değişkenleri bulalım
    binary_cols = [col for col in cat_cols if dataframe[col].dtype not in ["int64", "float64"]
                   and dataframe[col].nunique() == 2]
    return binary_cols
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins="auto")
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# Hedef Değişkenin Kategorik Değişkenler ile Analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    """
     Computes and prints the mean of the target variable for each category in the given categorical column.
    :param dataframe: The input DataFrame containing the data.
    :param target: The name of the target variable column.
    :param categorical_col: The name of the categorical column for which to compute the summary.
    :return:
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col, observed=True)[target].mean()}))

def detail_target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    """
    Computes and prints the mean of the target variable, count, and ratio for each category in the given categorical column.

    :param dataframe: The input DataFrame containing the data.
    :param target: The name of the target variable column.
    :param categorical_col: The name of the categorical column for which to compute the summary.
    :return:
    """
    print("Analyzed variable name::" + categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

    if plot:
        plt.figure(figsize=(10, 10))
        sns.countplot(x=dataframe[categorical_col], data=dataframe, color="red")
        plt.title(categorical_col)
        plt.xticks(rotation=30)
        plt.show(block=True)

def target_summary_with_cat_crosstab(dataframe, target, categorical_col):
    """
    iki veya daha fazla kategorik değişken arasındaki ilişkiyi ve dağılımı gösteren bir tablo oluşturmak
    için kullanılan bir fonksiyondur. Bu fonksiyon kategorik verilerin yüzdelerini hesaplar.
    """
    crosstab = pd.crosstab(dataframe[categorical_col], dataframe[target], normalize='index')
    # Sonuçları yazdırma
    print(f"\n{categorical_col} değişkenine göre {target} dağılımı:")
    print(crosstab)
    print("\n")

# Hedef Değişkenin Sayısal Değişkenler ile Analizi
def target_summary_with_num(dataframe, target, numeric_col):
    print("Analyzed variable name::" + numeric_col)
    print(pd.DataFrame(dataframe.groupby(target, observed=True).agg({numeric_col: "mean"})))
    print("=====================================================")
