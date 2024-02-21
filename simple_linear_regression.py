import numpy as np

class SimpleLinearRegression():
    def __init__(self):
        self.coefficient = None
        self.intercept = None

    def fit(self,x,y):
        '''
            Input is 1 feature and 1 target column. Compute intercept and coefficient
        '''
        self.coefficient = self._estimate_coefficient(x,y)
        self.intercept = self._estimate_intercept(x,y)

    def _estimate_coefficient(self,x,y):
        '''
            β1 = Σ{i=0}^{n}(x_i-x_bar)(y_i-y_bar)/Σ{i=0}^{n}(x_i-x_bar)^2
        '''
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        num = 0
        den = 0
        for i in range(x.shape[0]):
            num += (x[i]-x_bar)*(y[i]-y_bar)
            den += (x[i]-x_bar)**2

        return num/den

    def _estimate_intercept(self,x,y):
        '''
           β0 = y_bar - β1*x_bar 
        '''

        x_bar = np.mean(x)
        y_bar = np.mean(y)

        return y_bar - (self.coefficient)*(x_bar)

    def predict(self,x):
        '''
           y_i =  β1*x_i + β0
        '''
        y_pred = []
        for i in range(x.shape[0]):
            y_pred.append(self.coefficient*x[i]+self.intercept)

        return y_pred

    def r2_score(self,y_test,y_pred):
        '''
            r2 = 1 - (rss/tss)
            rss =  Σ{i=0}^{n}(y_true-y_pred)^2
            tss =  Σ{i=0}^{n}(y_true-y_bar)^2
        '''
        y_bar_test = np.mean(y_test)
        rss = 0
        tss = 0
        for i in range(y_pred.shape[0]):
            rss += (y_test[i]-y_pred[i])**2
            tss += (y_test[i]-y_bar_test)**2
        return 1-(rss/tss)









