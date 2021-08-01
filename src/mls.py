import numpy as np
import numexpr as ne

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def grad_sigmoid(z):
    if type(z) != np.ndarray:
        raise TypeError

    sigma_z = sigmoid(z)
    g = sigma_z*(1-sigma_z)
    return g

def randominit_weights(row, col):
    return np.random.rand(row,col)*2*0.12-0.12
    
def feature_normalize(X):
    if type(X) != np.ndarray:
        raise TypeError

    X_norm = np.zeros(X.shape)
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))

    mu = np.atleast_2d(np.mean(X, axis=0))
    sigma = np.atleast_2d(np.std(X, axis=0))
    X_norm = (X-mu)/sigma

    return (X_norm, mu, sigma)

# =============================================================================================================================================================

class GradientBase:
    def __init__(self, X, y):
        self._X = X
        self._y = y
        self._cost_history = 0
        self._w = 0
        self._b = 0
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, X):
        if type(X) != np.ndarray:
            raise TypeError
        self._X = np.atleast_2d(X)
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y):
        if type(y) != np.ndarray:
            raise TypeError
        self._y = np.atleast_2d(y)
    
    @property
    def cost_history(self):
        return self._cost_history
    
    @cost_history.setter
    def cost_history(self):
        pass

    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, w):
        self._w = w

    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        self._b = b

    def shuffle_dataset(self):
        if self._X.shape[0] != self._y.shape[0] or self._X is None or self._y is None:
            raise ValueError
        index = np.arange(self._X.shape[0])
        np.random.shuffle(index)
        self._X = self._X[index,:]
        self._y = self._y[index,:]
    
    def adam(self, num_iters, y=None, batch=1, lmbda=0.1, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10e-8):
        """
            diffGrad
        """
        if y is None:
            y = self._y
        if batch == 1:
            lmbda = 0
        elif batch < 1:
            raise ValueError
        X = self._X
        m = X.shape[0]
        w = self._w
        b = self._b
        m_dw = np.zeros(w.shape)
        m_db = np.zeros(b.shape)
        v_dw = np.zeros(w.shape)
        v_db = np.zeros(b.shape)
        t = 0
        index = 0
        cost = np.zeros((num_iters,1))
        dw = w
        db = b
        while True:
            t += 1
            if index >= X.shape[0]:
                index = 0
            
            prev_dw = dw
            prev_db = db

            dw,db = self.grad_function(w, b, X[index:(index+batch),:], y[index:(index+batch),:], m, lmbda)
            m_dw = beta1*m_dw + (1-beta1)*dw
            m_db = beta1*m_db + (1-beta1)*db

            v_dw = beta2*v_dw + (1-beta2)*np.power(dw,2)
            v_db = beta2*v_db + (1-beta2)*np.power(db,2)

            m_dw_c = m_dw/(1-np.power(beta1,t))
            m_db_c = m_db/(1-np.power(beta1,t))

            v_dw_c = v_dw/(1-np.power(beta2,t))
            v_db_c = v_db/(1-np.power(beta2,t))
            
            xi_dw = sigmoid(np.abs(prev_dw-dw))
            xi_db = sigmoid(np.abs(prev_db-db))

            w = w - (alpha * xi_dw * m_dw_c)/(np.sqrt(v_dw_c)+epsilon)
            b = b - (alpha * xi_db * m_db_c)/(np.sqrt(v_db_c)+epsilon)

            cost[t-1,0] = self.cost_function(w, b, X, y, m, lmbda)

            if t == num_iters:
                break

            index += batch

        self._w = w
        self._b = b
        self._cost_history = cost
        
        return (w,b,cost)

# =============================================================================================================================================================

class LogisticRegression(GradientBase):
    def __init__(self, X, y):
        super(LogisticRegression,self).__init__(X,y)
        self._w = randominit_weights(X.shape[1],1)
        self._b = randominit_weights(1,1)

    def cost_function(self, w, b, x, y, m, lmbda):
        J = 0
        h_theta = sigmoid(x.dot(w) + b)

        J = -1/x.shape[0]*( y.T.dot(np.log(h_theta)) + (1-y).T.dot(np.log(1-h_theta)) ) + lmbda/(2*m) * w.T.dot(w)

        return J
    
    def grad_function(self, w, b, x, y, m, lmbda):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        dw = np.zeros(w.shape)
        h_theta = sigmoid(x.dot(w) + b)

        dw = 1/x.shape[0]*x.T.dot(h_theta - y) + lmbda/m*w
        db = np.mean(h_theta - y)

        return (dw, db)

    def predict(self, X):
        m = X.shape[0]
        p = np.zeros((m,1))

        p = sigmoid(X.dot(self._w) + self._b) >= 0.8

        return p
    
    def predict_onevsall(self, X):
        m = X.shape[0]
        p = np.zeros((m,2))

        h = sigmoid(X.dot(self._w) + self._b)
        p[:,1] = np.atleast_2d(h.argmax(axis=1))
        p[:,0] = np.atleast_2d(h.max(axis=1))

        return p
    
    def calculate_training_test_cost(self, X_train, y_train, X_test, y_test, lmbda=0):
        if type(X_test) != np.ndarray or type(y_test) != np.ndarray:
            raise ValueError
        m = X_test.shape[0] if X_test.shape[0] < X_train.shape[0] else X_train.shape[0]
        train_cost = np.zeros((m,1))
        test_cost = np.zeros((m,1))
        for i in range(m):
            train_cost[i] = self.cost_function(self._w,self._b,X_train[:i+1,:],y_train[:i+1,:],m,lmbda)
            test_cost[i] = self.cost_function(self._w,self._b,X_test[:i+1,:],y_test[:i+1,:],m,lmbda)
        return (train_cost,test_cost)

# =============================================================================================================================================================

class NeuralNetwork(GradientBase):
    """ 
        1 hidden layer ANN with 25 activation units
    """
    def __init__(self, input_layer_size, hidden_layer_size, num_labels, X, y):
        super(NeuralNetwork,self).__init__(X,y)
        w1 = randominit_weights(input_layer_size,hidden_layer_size)
        w2 = randominit_weights(hidden_layer_size,num_labels)
        self._w = np.concatenate((w1,w2),axis=None)
        b1 = randominit_weights(1,hidden_layer_size)
        b2 = randominit_weights(1,num_labels)
        self._b = np.concatenate((b1,b2),axis=None)
        self._input_layer_size = input_layer_size
        self._hidden_layer_size = hidden_layer_size
        self._num_labels = num_labels

    def cost_function(self, params_w, params_b, x, y, m, lmbda):
        w1 = params_w[:self._input_layer_size * self._hidden_layer_size].reshape((self._input_layer_size,self._hidden_layer_size))
        w2 = params_w[self._input_layer_size * self._hidden_layer_size:].reshape((self._hidden_layer_size,self._num_labels))
        b1 = np.atleast_2d(params_b[:self._hidden_layer_size])
        b2 = np.atleast_2d(params_b[self._hidden_layer_size:])
        J = 0
        a2 = sigmoid(x.dot(w1) + b1)
        a3 = sigmoid(a2.dot(w2) + b2)
        h_theta = a3

        J = -1/x.shape[0] * np.sum(np.sum( y * np.log(h_theta) + (1-y) * np.log(1-h_theta) ))
        J += lmbda/(2*m) * ( sum(sum( np.power(w1,2) )) + sum(sum( np.power(w2,2) )) )

        return J
    
    def grad_function(self, params_w, params_b, x, y, m, lmbda):
        m = x.shape[0]
        w1 = params_w[:self._input_layer_size * self._hidden_layer_size].reshape((self._input_layer_size,self._hidden_layer_size))
        w2 = params_w[self._input_layer_size * self._hidden_layer_size:].reshape((self._hidden_layer_size,self._num_labels))
        b1 = np.atleast_2d(params_b[:self._hidden_layer_size])
        b2 = np.atleast_2d(params_b[self._hidden_layer_size:])

        a1 = x
        z2 = a1.dot(w1)
        a2 = sigmoid(z2 + b1)
        z3 = a2.dot(w2)
        a3 = sigmoid(z3 + b2)
        
        delta3 = a3 - y
        delta2 = delta3.dot(w2.T) * grad_sigmoid(z2)

        w1 = 1/m * (delta2.T.dot(a1)).T
        w2 = 1/m * (delta3.T.dot(a2)).T
        b1 = np.mean(delta2,axis=0)
        b2 = np.mean(delta3,axis=0)

        w1 += lmbda/m * w1
        w2 += lmbda/m * w2

        params_w = np.concatenate((w1,w2),axis=None)
        params_b = np.concatenate((b1,b2),axis=None)

        return (params_w,params_b)

    def predict(self, X):

        w1 = self._w[:self._input_layer_size * self._hidden_layer_size].reshape((self._input_layer_size,self._hidden_layer_size))
        w2 = self._w[self._input_layer_size * self._hidden_layer_size:].reshape((self._hidden_layer_size,self._num_labels))
        b1 = np.atleast_2d(self._b[:self._hidden_layer_size])
        b2 = np.atleast_2d(self._b[self._hidden_layer_size:])

        h1 = sigmoid(X.dot(w1) + b1)
        h2 = sigmoid(h1.dot(w2) + b2)

        p = np.zeros((X.shape[0],2))
        p[:,1] = np.atleast_2d(h2.argmax(axis=1))
        p[:,0] = np.atleast_2d(h2.max(axis=1))

        return p
    
    def calculate_training_test_cost(self, X_train, y_train, X_test, y_test, lmbda=0):
        if type(X_test) != np.ndarray or type(y_test) != np.ndarray:
            raise ValueError
        m = X_test.shape[0] if X_test.shape[0] < X_train.shape[0] else X_train.shape[0]
        train_cost = np.zeros((m,1))
        test_cost = np.zeros((m,1))
        for i in range(m):
            train_cost[i] = self.cost_function(self._w,self._b,X_train[:i+1,:],y_train[:i+1,:],m,lmbda)
            test_cost[i] = self.cost_function(self._w,self._b,X_test[:i+1,:],y_test[:i+1,:],m,lmbda)
        return (train_cost,test_cost)

# =============================================================================================================================================================

class SupportVectorMachine:
    def __init__(self, X, y, kernel='linear'):
        self._X = X
        self._y = y
        self._m = X.shape[0]
        self._w = 0.
        self._b = 0.
        self._kernel = kernel
        self._sigma = 0.
        self._C = 0.
        self._tol = 0.
        self._alpha = np.atleast_2d([0.]*X.shape[0]).T
        self._E = np.atleast_2d([0.]*X.shape[0]).T
        self._K = None
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, X):
        if type(X) != np.ndarray:
            raise TypeError
        self._X = np.atleast_2d(X)
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y):
        if type(y) != np.ndarray:
            raise TypeError
        self._y = np.atleast_2d(y)

    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, w):
        self._w = w

    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        self._b = b
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, K):
        if type(K) != np.ndarray:
            raise TypeError
        self._K = K
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    def shuffle_dataset(self):
        if self._X.shape[0] != self._y.shape[0] or self._X is None or self._y is None:
            raise ValueError
        index = np.arange(self._X.shape[0])
        np.random.shuffle(index)
        self._X = self._X[index,:]
        self._y = self._y[index,:]
    
    def kernel(self, x1, x2):
        def linear(self, x1, x2):
            return x1.T.dot(x2)
        def gaussian(self, x1, x2):
            return np.exp( -( np.power(np.linalg.norm(x1-x2,axis=1),2) / (2 * np.power(self._sigma,2)) ) )
        kernel_func = None
        if self._kernel == 'linear':
            kernel_func = linear
        elif self._kernel == 'gaussian':
            kernel_func = gaussian
        return kernel_func(self, np.atleast_2d(x1), np.atleast_2d(x2))
    
    def _f(self, x):
        return ((self._alpha * self._y).T.dot(self.kernel(self._X,x)) + self._b).item()
    
    def calculate_K(self, sigma=0.1):
        self._sigma = sigma
        if self._kernel == 'linear':
            K = self._X.dot(self._X.T)
        elif self._kernel == 'gaussian':
            X_norm = np.sum(self._X ** 2, axis = -1)
            K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                    'A' : X_norm[:,None],
                    'B' : X_norm[None,:],
                    'C' : np.dot(self._X, self._X.T),
                    'g' : 1/(2*np.power(self._sigma,2)),
            })
        self._K = K
    
    def smo(self, y=None, max_passes=5, sigma=0.1, C=1, tol=1e-3):
        if y is not None:
            self._y = y
        self._y[self._y == 0] = -1
        self._C = C
        self._tol = tol
        if sigma is not None:
            self.calculate_K(sigma)
        L = 0.
        H = 0.
        eta = 0.
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(self._m):
                self._E[i] = self._b + (self._alpha * self._y).T.dot(np.atleast_2d(self._K[:,i]).T) - self._y[i]
                if not( (self._y[i]*self._E[i] < -tol and self._alpha[i] < C) or (self._y[i]*self._E[i] > tol and self._alpha[i] > 0) ):
                    continue

                j = np.ceil((self._m - 1) * np.random.uniform())
                while j == i:
                    j = np.ceil((self._m - 1) * np.random.uniform())
                j = int(j)
                
                self._E[j] = self._b + (self._alpha * self._y).T.dot(np.atleast_2d(self._K[:,j]).T) - self._y[j]
                alpha_i_old = self._alpha[i]
                alpha_j_old = self._alpha[j]

                L = np.maximum(0, self._alpha[j] + self._alpha[i] - self._C) if self._y[i] == self._y[j] else np.maximum(0, self._alpha[j] - self._alpha[i])
                H = np.minimum(self._C, self._alpha[j] + self._alpha[i]) if self._y[i] == self._y[j] else np.minimum(self._C, self._C + self._alpha[j] - self._alpha[i])
                
                if L == H:
                    continue
                
                eta = 2 * self._K[i,j] - self._K[i,i] - self._K[j,i]
                if eta >= 0:
                    continue

                self._alpha[j] = self._alpha[j] - (self._y[j] * (self._E[i] - self._E[j])) / eta

                self._alpha[j] = np.minimum(H, self._alpha[j])
                self._alpha[j] = np.maximum(L, self._alpha[j])

                if abs(self._alpha[j] - alpha_j_old) < tol:
                    self._alpha[j] = alpha_j_old
                    continue

                self._alpha[i] = self._alpha[i] + self._y[i] * self._y[j] * (alpha_j_old - self._alpha[j])

                b1 = self._b + self._E[i] \
                    - self._y[i] * (self._alpha[i] - alpha_i_old) * self._K[i,j] \
                    - self._y[j] * (self._alpha[j] - alpha_j_old) * self._K[i,j]
                b2 = self._b + self._E[j] \
                    - self._y[i] * (self._alpha[i] - alpha_i_old) * self._K[i,j] \
                    - self._y[j] * (self._alpha[j] - alpha_j_old) * self._K[j,j]
                
                if 0 < self._alpha[i] and self._alpha[i] < self._C:
                    self._b = b1
                elif 0 < self._alpha[j] and self._alpha[j] < self._C:
                    self._b = b2
                else:
                    self._b = (b1 + b2) / 2
                
                num_changed_alphas += 1

            passes = (passes + 1) if num_changed_alphas == 0 else 0
        idx = self._alpha > 0
        idx = idx.ravel()
        self._X = self._X[idx,:]
        self._y = self._y[idx,:]
        self._alpha = self._alpha[idx,:]
        self._w = (self._alpha * self._y).T.dot(self._X).T
        self._m = self._X.shape[0]
        
        return (self._w,self._b)
    
    def predict(self, X):
        p = np.zeros((X.shape[0],2))
        pred = np.zeros((X.shape[0],1))
        p[:,0] = np.array([ self._f(X[i,:]) for i in range(X.shape[0]) ])
        pred[p[:,0] >= 0] = 1
        pred[p[:,0] < 0] = 0
        p[:,1] = pred.ravel()
        return p