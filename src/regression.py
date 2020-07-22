
import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy.linalg import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge



class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None


    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in range(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta.T: ", theta.T)
            yhat = X*theta
            theta = theta -  (X.T * (yhat - y)) * (self.alpha / n)
        return theta


    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost
        '''
        n,d = X.shape
        yhat = X*theta
        J =  (yhat-y).T * (yhat-y) / n
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        return J_scalar


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas Dataframe
            y is an n-dimensional Pandas Series
        '''
        n = len(y)
        X = X.to_numpy()
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term

        y = y.to_numpy()
        n,d = X.shape
        y = y.reshape(n,1)

        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))

        self.theta = self.gradientDescent(X,y,self.theta)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas DataFrame
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        X = X.to_numpy()
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        return pd.DataFrame(X*self.theta)






class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1e-8, tuneLambda = False, regLambdaValues = []):
        self.degree = degree
        self.regLambda = regLambda
        self.alpha = 0.25
        self.theta = None
        self.tuneLambda = tuneLambda
        self.regLambdaValues = regLambdaValues



    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.
        Returns:
            A n-by-d data frame, with each column comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.
        Arguments:
            X is an n-by-1 data frame
            degree is a positive integer
        '''
        X = X.copy()
        for i in range(1, degree ):
          X[i] = X[0].apply(lambda x: pow(x, i + 1))
        return X

    def computeCost(self, X, y, theta):
      '''
      Computes the objective function
      Arguments:
        X is a n-by-d numpy matrix
        y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost
        '''
      X = X.copy()
      y = y.copy()
      n,d = X.shape
      yhat = X*theta
      J =  (yhat-y).T * (yhat-y) / n + self.regLambda * theta[1:].T * theta[1:]
      J_scalar = J.tolist()[0][0]  # convert matrix to scalar
      return J_scalar

    def findOptimalLambda(self, X, y):
      X = X.copy()
      y = y.copy()
      clf = Ridge(solver='lsqr')
      cv = GridSearchCV(estimator=clf, param_grid={'alpha':self.regLambdaValues}, scoring="r2", cv=2)
      cv.fit(X,y)
      return cv.best_params_.get('alpha')


    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        X = X.copy()
        y = y.copy()
        n,d = X.shape
        self.JHist = []
        prev_theta = theta
        converge = 1 + 0.0001
        i = 0
        while True:
            if (converge < 0.0001):
              break
            cost = self.computeCost(X, y, prev_theta)
            self.JHist.append((cost, prev_theta))
            yhat = X * prev_theta
            print((X.T * (yhat - y)))
            new_theta = prev_theta - (X.T * (yhat - y)) * (self.alpha / n)
            new_theta[1:] = new_theta[1:] - (self.alpha *self.regLambda)*new_theta[1:]
            theta_diff = new_theta - prev_theta
            if i % 1000 == 0:
              print("Converge: ", converge, " Cost: ", self.JHist[i][0], " Theta.T: ", prev_theta.T)
            converge = np.linalg.norm(theta_diff)
            prev_theta = new_theta
            i+=1
        self.theta = prev_theta

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 data frame
                y is an n-by-1 data frame
            Returns:
                No return value
        '''
        X = X.copy()
        y = y.copy()
        if self.tuneLambda:
          self.regLambda = self.findOptimalLambda(X, y)

        X = self.polyfeatures(X = X, degree = self.degree)
        n = len(y)
        X = X.to_numpy()
        self.mean = X.mean(axis = 0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean) / self.std
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        y = y.to_numpy()
        n,d = X.shape
        y = y.reshape(n,1)

        self.theta = np.matrix(np.zeros((d,1)))

        self.gradientDescent(X,y,self.theta)



    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 data frame
        Returns:
            an n-by-1 data frame of the predictions
        '''
        # TODO
        X = X.copy()
        X = self.polyfeatures(X = X, degree = self.degree)
        X = (X - self.mean) / self.std

        X = np.c_[np.ones((X.shape[0],1)), X]     # Add a row of ones for the bias term
        return pd.DataFrame(X*self.theta)
