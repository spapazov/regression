import numpy as np
import matplotlib.pyplot as plt

def test_polyreg_univariate():
    '''
        Test polynomial regression
    '''

    # load the data
    filepath = "./test.txt"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]



    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree=d, tuneLambda=True, regLambdaValues= [ 0.001, 0.003, 0.001])
    model.fit(X, y)

    # output predictions
    xpoints = pd.DataFrame(np.linspace(np.max(X), np.min(X), 100))
    ypoints = model.predict(xpoints)
    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = '+str(d))
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print(model.theta)

    #  # # Compute the closed form solution
    # n,d = X.shape
    # X = np.asmatrix(X.to_numpy())
    # X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
    # y = np.asmatrix(y.to_numpy())

    # y = y.reshape(n,1)
    # thetaClosedForm = inv(X.T*X)*X.T*y
    # print("thetaClosedForm: ", thetaClosedForm.T)

test_polyreg_univariate()
