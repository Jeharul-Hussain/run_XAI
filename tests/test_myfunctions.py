import sklearn.ensemble
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from XAIM import myfunctions


boston = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

#xgbr = sklearn.ensemble.GradientBoostingRegressor()
#xgbr.fit(X_train, y_train)

#svr = SVR(kernel='rbf')
#svr.fit(X_train, y_train)

myfunctions.treeMLAlgosExplainer(rfr, X_train, X_test, "", 2)