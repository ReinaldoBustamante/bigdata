import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
import time

start = time.time()
iris = datasets.load_breast_cancer()
X = iris.data[:, :2][:569]
Y = iris.target[:569]

mymodel = LogisticRegression(C=1e5)
mymodel.fit(X, Y)

_, ax = plt.subplots(figsize=(6, 4))
DecisionBoundaryDisplay.from_estimator(
    mymodel,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    eps=0.5,
    xlabel= 'radius',
    ylabel= 'texture'
)

plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Paired)
plt.savefig("log_model.png")

end=time.time()

total = end - start

print("Time elapsed: ",end="")
print(total)