import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random as rd
start = time.time()

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

df = pd.read_csv(url, names=['variance','skewness','curtosis','entropy','target'], nrows=1372) #Cambiar nrows para modificar cantidad datos.

features = ['variance','skewness','curtosis','entropy']
x = df.loc[:, features].values
#print(x)
#print(y)

y = df.loc[:,['target']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

plt.figure(figsize=(8, 8))

targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.grid()
plt.legend(targets)
plt.savefig('pca_model.png')

end=time.time()

total = end - start

print("Time elapsed: ",end="")
print(total)
