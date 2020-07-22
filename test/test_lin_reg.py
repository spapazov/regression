"""### Test code for linear regression"""

def test_linreg(n_iter = 2000):
  # load the data
  filepath = "./test.txt"
  df = pd.read_csv(filepath, header=None)

  X = df[df.columns[:-1]]
  y = df[df.columns[-1]]

  n,d = X.shape

  # # Standardize features
  from sklearn.preprocessing import StandardScaler
  standardizer = StandardScaler()
  X = pd.DataFrame(standardizer.fit_transform(X))  # compute mean and stdev on training set for standardization

  # # initialize the model
  init_theta = np.matrix(np.random.randn((d+1))).T
  alpha = 0.01

  # # Train the model
  lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
  lr_model.fit(X,y)

  # # Compute the closed form solution
  X = np.asmatrix(X.to_numpy())
  X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
  y = np.asmatrix(y.to_numpy())
  n,d = X.shape
  y = y.reshape(n,1)
  thetaClosedForm = inv(X.T*X)*X.T*y
  print("thetaClosedForm: ", thetaClosedForm.T)

"""# Run the Linear Regression Test"""

test_linreg(2000)
