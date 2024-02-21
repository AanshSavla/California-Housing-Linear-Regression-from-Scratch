import numpy as np
import copy

class MultipleLinearRegression():
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, x, y):
        # Prepare X and Y values for coefficient estimates
        x = self._transform_x(x)
        y = self._transform_y(y)

        betas = self._estimate_coefficients(x,y)

        #intercept becomes a vector of ones
        self.intercept = betas[0]

        # coefficients becomes the rest of the betas
        self.coefficients = betas[1:]

    def predict(self, x):
        '''
            y = b_0 + b_1*x + ... + b_i*x_i
        '''
        predictions = []
        for row in x:

            pred = np.multiply(row, self.coefficients)
            pred = sum(pred)
            pred += self.intercept

            predictions.append(pred)

        return predictions

    def r2_score(self, y_true, y_pred):
        '''
            r2 = 1 - (rss/tss)
            rss = sum_{i=0}^{n} (y_i - y_hat)^2
            tss = sum_{i=0}^{n} (y_i - y_bar)^2
        '''
        y_values = y_true
        y_average = np.average(y_values)

        residual_sum_of_squares = 0
        total_sum_of_squares = 0

        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i])**2
            total_sum_of_squares += (y_values[i] - y_average)**2

        return 1 - (residual_sum_of_squares/total_sum_of_squares)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        x = np.hstack((np.ones((x.shape[0],1)),x))
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y

    def _estimate_coefficients(self, x ,y):
        '''
            β = (X^T X)^-1 X^T y
            Estimates both the intercept and all coefficients
        '''
        xT = x.transpose()

        inversed = np.linalg.inv(xT.dot(x))
        coefficients = inversed.dot(xT).dot(y)

        return coefficients
