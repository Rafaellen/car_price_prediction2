import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def processing_data(df):
    df = df.drop(columns=["Unnamed: 0", "Name", "New_Price"], errors="ignore")

    # conversão str -> número
    def convert_str(value):
        try:
            return float(str(value).split()[0])
        except:
            return None

    df['Mileage'] = df['Mileage'].apply(convert_str)
    df['Engine'] = df['Engine'].apply(convert_str)
    df['Power'] = df['Power'].apply(convert_str)

    df = df.dropna()

    # One-hot encoding para variáveis categoricas
    df = pd.get_dummies(df, columns=[
                        "Location", "Fuel_Type", "Transmission", "Owner_Type"], drop_first=True)

    X = df.drop(columns=["Price"])
    Y = df["Price"]

    return X, Y


def normal_data(X):
    scaler = MinMaxScaler()
    x_normal = scaler.fit_transform(X)
    return x_normal, scaler
