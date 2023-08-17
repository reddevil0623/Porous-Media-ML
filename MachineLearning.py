import matplotlib.pyplot as plt
import pickle
import numpy as np

# import data set
X_train = pickle.load(open('porous_rock_images_train.pkl', 'rb'))
y_train=pickle.load(open('flux_train.pkl','rb'))
X_test = pickle.load(open('porous_rock_images_test.pkl', 'rb'))
y_test=pickle.load(open('flux_test.pkl','rb'))

# calculate the persistent homology
cubical_persistence0 = CubicalPersistence(homology_dimensions=[0],n_jobs=-1)
cubical_persistence1 = CubicalPersistence(homology_dimensions=[1],n_jobs=-1)
Cubical_train0 = cubical_persistence0.fit_transform(X_train)
Cubical_train1 = cubical_persistence1.fit_transform(X_train)

X_signed = signed_distance_function(X_train, 1, False)
cubical_signed0 = cubical_persistence0.fit_transform(X_signed)
cubical_signed1 = cubical_persistence1.fit_transform(X_signed)

X_transformed = AddColumn(X_signed,-10000)
cubical_transformed0 = cubical_persistence0.fit_transform(X_transformed)

Cubical_test0 = cubical_persistence0.fit_transform(X_test)
Cubical_test1 = cubical_persistence1.fit_transform(X_test)

X_test1 = signed_distance_function(X_test, 1, False)
Cubical_test2 = cubical_persistence0.fit_transform(X_test1)
Cubical_test3 = cubical_persistence1.fit_transform(X_test1)
Cubical_test4 = cubical_persistence0.fit_transform(AddColumn(X_test1,-10000))

# calculating feature vectors for the training set
SB_train0 = PersistenceSum(Cubical_train0) # one can also apply these functions to the Cubical_signed0 and Cubical_signed1 to get persistent barcodes of the signed distance function filtration.
SB_train1 = PersistenceSum(Cubical_train1)
AF_train0 = AlgebraicFunctions(Cubical_train0)
AF_train1 = AlgebraicFunctions(Cubical_train1)
PL_train0 = Landscape(Cubical_train0)
PL_train1 = Landscape(Cubical_train1)
SS_train = SaddleSum(Cubical_train0,Cubical_train1)
QS_train0 = quadrant_separation(Cubical_train0)
QS_train1 = quadrant_separation(Cubical_train1)
ConVal = Connection(cubical_transformed0)

Sum = Concatenation(SB_train0, SB_train1) # using concatenation function to form the feature vectors. We illustrate the case when using sum of barcodes only, but one can iterately using this function to form any combination of vectorizations.

# Normalizations for the training set, note that we need to normalize the feature vectors by each coordinate before fitting into the machine learning models.
Sum = Concatenation(SumOriginal,SumSigned)
Means = GetMean(Sum)
Sd = GetSd(Sum)
Normalized = ColumnNormalize(Sum, GetMean(Sum), GetSd(Sum))
y_train = np.array(y_train)
y_sd = np.std(y_train)
y_normalized = y_train/y_sd # subtracting mean will not make a difference here.

# Feature vectors and normalizations for the testing set
SB_test0 = PersistenceSum(Cubical_test0)
SB_test1 = PersistenceSum(Cubical_test1)
AF_test0 = AlgebraicFunctions(Cubical_test0)
AF_test1 = AlgebraicFunctions(Cubical_test1)
PL_test0 = Landscape(Cubical_test1)
PL_test1 = Landscape(Cubical_test1)
SS_test = SaddleSum(Cubical_test0,Cubical_test1)
QS_test0 = quadrant_separation(Cubical_test0)
QS_test1 = quadrant_separation(Cubical_test1)
ConVal1 = Connection(Cubical_test4)

S = Concatenation(SB_test0,SB_test1)
X_test_tda = ColumnNormalize(S, Means, Sd) # we need to use the mean and standard deviation of the training set to normalize the testing set as well.

# create linear regression  model and use it to make predictions
model = linear_model.LinearRegression()
model.fit(Normalized, y_normalized)
y_pred = model.predict(X_test_tda)
y_test = np.array(y_test)
y_test = y_test/y_sd

# printing report for linear regression
# The coefficients
print("Coefficients: \n", model.coef_)
# The mean squared error
print("Mean squared error: %.4f" % mean_squared_error(y_test, y_pred))
# The R2 score: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# create Ridge or Lasso Regression training model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
model = Ridge() # or lasso = Lasso() to use the Lasso regression
sequence = [ ]
for i in range(-10, 10):
    sequence.append(10**i)
param = {'alpha': sequence}
search = GridSearchCV(estimator=model, param_grid=param, scoring='r2',cv=5)
search.fit(Normalized,y_train)
print(search.best_params_)
print(search.best_score_)

# building model based on GridSearchCV and make predictions
model = Lasso(alpha=0.001) # manually tune this parameter based on the GridsearchCV result
model.fit(Normalized,y_normalized)
y_pred = model.predict(X_test_tda)

# printing report for Ridge or Lasso regression
# The coefficients
print("Coefficients: \n", model.coef_)
# The mean squared error
print("Mean squared error: %.4f" % mean_squared_error(y_test, y_pred))
# The R2 score: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
