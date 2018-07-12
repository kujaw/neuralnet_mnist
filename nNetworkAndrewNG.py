import numpy as np
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import loadmat

class NN_MLstanford(object):

    def __init__(self, reg_lambda=0, epsilon_init=0.12, hidden_layer_size=25, opti_method='TNC', maxiter=500):
        self.reg_lambda = reg_lambda
        self.epsilon_init = epsilon_init
        self.hidden_layer_size = hidden_layer_size
        self.method = opti_method
        self.maxiter = maxiter
        self.activation_func = self.sigmoid
        self.activation_func_deriv = self.sigmoid_deriv

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z):
        output = self.sigmoid(z)
        return output * (1 - output)

    def sumsqr(self, z):
        return np.sum(z ** 2)

    def rand_init(self, x, y):
        return np.random.rand(y, x+1) * 2 * self.epsilon_init - self.epsilon_init

    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1),t2.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2

    def _forward(self, X, t1, t2):
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1,)
        else:
            ones = np.ones(m).reshape(m,1)

        a1 = np.hstack((ones,X))

        z2 = np.dot(a1, t1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2))

        z3 = np.dot(a2, t2.T)
        a3 = self.activation_func(z3)
        return a1, z2, a2, z3, a3

    def cost_func(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        _, _, _, _, h = self._forward(X, t1, t2)
        term1 = -Y * np.log(h)
        term2 = (1 - Y) * np.log(1 - h)
        cost = term1 - term2
        J = np.sum(cost) / m

        if reg_lambda != 0:
            t1_reg = t1[:,1:]
            t2_reg = t2[:,1:]
            reg = (self.reg_lambda / (2 * m)) * (self.sumsqr(t1_reg) + self.sumsqr(t2_reg))
            J = J + reg
        return J

    def cost_func_deriv(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]
        t1_reg = t1[:,1:]
        t2_reg = t2[:,1:]

        Delta1, Delta2 = 0, 0

        a1, z2, a2, z3, a3 = self._forward(X, t1, t2)
        d3 = a3 - Y
        d2 = np.dot(d3, t2_reg) * self.activation_func_deriv(z2)

        Delta1 = np.dot(d2.T, a1)
        Delta2 = np.dot(d3.T, a2)

        Theta1_grad = Delta1 / m
        Theta2_grad = Delta2 / m

        if reg_lambda != 0:
            Theta1_grad[:,1:] = Theta1_grad[:,1:] + (reg_lambda / m) * t1_reg
            Theta2_grad[:,1:] = Theta2_grad[:,1:] + (reg_lambda / m) * t2_reg

        return self.pack_thetas(Theta1_grad, Theta2_grad)

    def fit(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas_0 = self.pack_thetas(theta1_0, theta2_0)

        options = {'maxiter': self.maxiter}
        _res = optimize.minimize(self.cost_func, thetas_0, jac=self.cost_func_deriv, method=self.method,
                                 args=(input_layer_size, self.hidden_layer_size, num_labels, X, y, 0), options=options)
        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)

    def predict(self, X):
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        _, _, _, _, h = self._forward(X, self.t1, self.t2)
        return h.T

### START SCRIPT
# Visualize a part of data
import displayData
print('Data visualization')
displayData.display()

print('Training model...')

data = loadmat('ex4data1.mat')
X, y = data['X'], data['y']
y = y.reshape(X.shape[0], )
y = y - 1

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Train neural network using train set, then predict on test set
nn = NN_MLstanford(maxiter=200)
nn.fit(X_train, y_train)

print('Done.')

# Predict
print('Predicting...')
accuracy = accuracy_score(y_test, nn.predict(X_test))
print('Accuracy score:', accuracy)
