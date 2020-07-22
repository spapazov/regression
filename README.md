# regression
Regression is a "from scratch" implementation of the linear and polynomial regression machine learning models. It is built
using only numpy and pandas and uses the idea of Gradient Descent to fit the models to an input data set.


## Background

Linear regression tries to fit a function ![formula](https://render.githubusercontent.com/render/math?math=H(X)=%20\theta_1x_1%20%2B%20\theta_2x_2%20%2B%20\theta_3x_3%20%2B%20...%2B\theta_nx_n%20$) to an input data set where the optimal ![formula](https://render.githubusercontent.com/render/math?math=\theta_1,\theta_2,\theta_3,...,\theta_n$) are to be determined. In order to do so, a loss function ![formula](https://render.githubusercontent.com/render/math?math=L(\Theta)$) is defined for the model ![formula](https://render.githubusercontent.com/render/math?math=H(X)$) over the training data ![formula](https://render.githubusercontent.com/render/math?math=(x_i,y_i)%20\in%20D$) such that:

![formula](https://render.githubusercontent.com/render/math?math=\Large%20L(\Theta)=\frac{1}{n}\sum_1^n%20[H(x_i)-y_i]^2$)

To find the optimal ![formula](https://render.githubusercontent.com/render/math?math=\theta_1,\theta_2,\theta_3,...,\theta_n$) we must apply gradient descent and repeatedly update the parameters until the loss function is brought to a local minima (i.e. the parameters achieve convergence). A single iteration of parameter updates is defined as:

![formula](https://render.githubusercontent.com/render/math?math=\Large\theta_j=\theta_j-\alpha\frac{\partial{L(\Theta)}}{\partial{\theta_j}},\small\alpha%20\in%20\mathbb{R}^{%2B}$)

These repeated parameter updates essentially take small incremental steps towards a minima of the loss function. Here is a visual depiction for a more intuitive sense:

![image](https://miro.medium.com/max/1400/1*yasmQ5kvlmbYMe8eDkyl6w.png)

When dealing with polynomial regression the same principles are applied however the input features are preprocessed such that: ![formula](https://render.githubusercontent.com/render/math?math=x_1=x_1,x_2=x_2^2,x_3=x_3^3,...,x_n=x_n^n$)
# Installation

Add clone the repo folder and add it to your project. Then import the regression file accordingly:

``` python
from regression.src import regression
```

# Usage

``` python
#Linear Regression
linear_regression = regression.LinearRegression()
linear_regression.fit(X,y) #X, y is the input as a pandas df
pred = linear_regression.predict(X) #X is input as a pandas df

#Polynomial Regression
poly_regression = regression.LinearRegression()
poly_regression.fit(X,y) #X, y is the input as a pandas df
pred = poly_regression.predict(X) #X is input as a pandas df


#Compute the value of the loss function for given parameters
# X, y and params to be provided as numpy vectors
cost_lin_reg = linear_regression.computeCost(X, y, params)
cost_poly_reg = poly_regression.computeCost(X, y, params)
```

# Test and Examples

An example test case is provided in the test folder
