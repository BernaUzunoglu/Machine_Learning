import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


# OUTLIERS
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


# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

# Aykırı değerleri hesaplanan eşik değeri ile değiştirme fonk.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# MISSING VALUE
# Eksik değere sahip olan değişkenlere ulaşalım
def missing_values_table(dataframe, na_name=False, na_cols_detail=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    na_columns_detail = {col: dataframe[col].isnull().sum() for col in dataframe.columns if dataframe[col].isnull().sum() > 0}

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name and na_cols_detail:
        return na_columns, na_columns_detail
    elif na_name:
        return na_columns
    elif na_cols_detail:
        return na_columns_detail
    else:
        return None

# ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        # Değişkenin tüm df deki oranını hesaplayalım
        tmp = temp_df[var].value_counts() / len(temp_df)
        # Verilen yüzdelik değerin altında olan değerleri bulalım.
        rare_labels = tmp[tmp < rare_perc].index
        # DF deki nadir olan değerleri Rare olarak sınıflayıp olmayanları olduğu gibi bırakalım
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def num_cols_standardization(dataframe, num_cols, scaler_type="ss"):
    """
    Verilen veriye göre belirtilen ölçekleyici tipini uygulayan bir fonksiyon.

    Args:
    scaler_type (str): Kullanılacak ölçekleyici tipi ('ss : Standart Scaler', 'rs : RobustScaler', 'mms : MinMax Scaler').
    data (pandas.DataFrame): Ölçeklemek istediğimiz veri seti.

    Returns:
    pandas.DataFrame: Ölçeklenmiş veri seti.
    """
    if scaler_type == 'ss':
        scaler = StandardScaler()
    elif scaler_type == 'rs':
        scaler = RobustScaler()
    elif scaler_type == 'mms':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Geçersiz ölçekleyici tipi. 'ss', 'rs' veya 'mms' olmalıdır.")

    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe
