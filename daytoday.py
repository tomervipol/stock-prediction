from keras.models import load_model
import pandas as pd
from sklearn import preprocessing
stock_name=input("enter stock name: ")
path=input("enter path: ")
model=load_model(path+"/"+stock_name+".h5")
df=pd.read_csv(path+"/"+stock_name+"22.csv")
df.drop(['Volume', 'Close',"Date"], 1, inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
X = df.as_matrix()
X = np.reshape(X, (1,X.shape[0], X.shape[1]))
p=model.predict(X)
def denormalize(normalized_value):
    df = pd.read_csv(path+"/"+stock_name+".csv", index_col=0)
    df = df['Adj Close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new
p=denormalize(p)
print(p)
