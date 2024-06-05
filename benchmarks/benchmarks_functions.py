import numpy as np
import joblib
import sys

sys.path.insert(0, '../machine_learning_models')



from sklearn.metrics import *
from bsplines import bSplines



class targets:
    
    def __init__(self, problem, domain=[]):
        
        self.problem = problem
        self.domain = domain
        
        if len(self.domain) == 0 and self.problem == 'oscillator':
            self.domain = np.linspace(0, 20*np.pi, 100)
            
        if len(self.domain) == 0 and self.problem == 'decay':
            self.domain = np.linspace(0, 1, 50)

        if len(self.domain) == 0 and self.problem == 'logistic_growth':
            self.domain = np.linspace(0, 10, 50)
        
        
    def oscillator(self, t):
        
        A = 4
        beta = 0.1
        gamma_ = 0.60
        phi = 12
        
        return A*np.exp(-beta*t)*np.sin(gamma_*t + phi)

    def logistic_growth(self, t):
        
        K = 200
        P0 = 1000
        r = 0.1
        
        return K/(1 + ((K-P0)/P0)*np.exp(-r*t))


    def decay(self, t):
        
        N0 = 11
        lambda_ = 3
        k = 12
        
        return N0*np.exp(-lambda_*t) + k
    
    def get_target(self):
        
        if self.problem == 'oscillator':
            
            target = self.oscillator(self.domain)
            
        if self.problem == 'decay':
            
            target = self.decay(self.domain)

            
        if self.problem == 'logistic_growth':
            
            target = self.logistic_growth(self.domain)
            
            
        if self.problem == 'emissivity_step':
            
            target = np.zeros((1,822))
            target[0][:475] = 1
            target = target[0]     
            
        if self.problem == 'emissivity_100':
        
            target = np.ones((1,822))
            target = target[0]     
            
        return target

class function:
    
    def __init__(self, problem, target, return_value=False):
        
        self.problem = problem
        self.target = target
        self.return_value = return_value
        
        if 'inconel' in self.problem:
            # print ('inconel')
            self.exp_model = joblib.load('../machine_learning_models/inconel_model.pkl')
            self.pca_model = joblib.load('../machine_learning_models/inconel_pca.pkl')

        else:
            # print ('ss')

            # self.exp_model = joblib.load('../models_benchmarks/inconel_model.pkl')
            self.exp_model = joblib.load('../machine_learning_models/ss_model.pkl')
            self.pca_model = joblib.load('../machine_learning_models/ss_pca.pkl')



        # self.exp_model = joblib.load('../models_benchmarks/forwardModel_8500_rf_older.pkl')
        # self.pca_model = joblib.load('../models_benchmarks/pca_50.pkl')

    def oscillator(self, x):
        
        A = x[0]
        beta = x[1]
        gamma_ = x[2]
        phi = x[3]
        t = np.linspace(0, 20*np.pi, 100)
        value = np.array([A*np.exp(-beta*t)*np.sin(gamma_*t + phi)])
        rmse = np.sqrt(mean_squared_error(self.target, value[0]))
        
        if self.return_value == True:
            return rmse, value
        else:
            return rmse
    
    def decay(self, x):
        
        N0 = x[0]
        lambda_ = x[1]
        k = x[2]
        t = np.linspace(0, 1, 50)
        value = np.array([N0*np.exp(-lambda_*t) + k])
        rmse = np.sqrt(mean_squared_error(self.target, value[0]))
            
        if self.return_value == True:
            return rmse, value
        else:
            return rmse

    def logistic_growth(self, x):
        
        K = x[0]
        P0 = x[1]
        r = x[2]
        t = np.linspace(0, 10, 50)
        
        value = np.array([K/(1 + ((K-P0)/P0)*np.exp(-r*t))])
        rmse = np.sqrt(mean_squared_error(self.target, value[0]))
        
        if self.return_value == True:
            return rmse, value
        else:
            return rmse
    
    def emissivity_ss(self, x):
        
        em_ = self.exp_model.predict([x])
        value = self.pca_model.inverse_transform(em_)
        # value = bSplines(em_, 'ss').generate_curves(em_)
        rmse = np.sqrt(mean_squared_error(self.target, value[0]))
        
        if self.return_value == True:
            return rmse, value
        else:
            return rmse
        
    def emissivity_inconel(self, x):
        
        em_ = self.exp_model.predict(np.array([x]))
        value = self.pca_model.inverse_transform(em_)
        # rmse = np.sqrt(mean_squared_error(self.target, value[0]))
        # value = bSplines(em_, 'inconel').generate_curves(em_)

        rmse = np.sqrt(mean_squared_error(self.target, value[0]))

        
        if self.return_value == True:
            return rmse, value
        else:
            return rmse
        
    def get_bounds(self):
        
        if self.problem == 'oscillator':
            
            #lb = np.array([2, 0.05, 0, 7])
            #ub = np.array([5, 0.2, 1, 15])
            # A = 4
            # beta = 0.1
            # gamma_ = 0.60
            # phi = 12
            
            lb = np.array([2, 0.05, 0, 3])
            ub = np.array([5, 0.4, 2, 15])
        
        elif self.problem == 'decay':
            
            lb = np.array([0, 0, 0])
            ub = np.array([20, 10, 20])

        elif self.problem == 'logistic_growth':
            
            # lb = np.array([100, 500, 0.01])
            # ub = np.array([600, 1500, 0.4])
            # K = 200
            # P0 = 1000
            # r = 0.1
            
            lb = np.array([100, 100, 0.01])
            ub = np.array([1200, 1400, 0.4])

        elif self.problem == 'emissivity_ss':
            
            lb = np.array([0.2, 10, 1])
            ub = np.array([1.3, 700, 42])

        elif self.problem == 'emissivity_inconel':
            
            lb = np.array([0.2, 10, 15])
            ub = np.array([1.3, 700, 28])
            
        return lb, ub
    
    def evaluate(self, x):
       
        if self.problem == 'oscillator':
            
            responses = self.oscillator(x)
         
        elif self.problem == 'decay':
            
            responses = self.decay(x)
            
        elif self.problem == 'logistic_growth':
            
            responses = self.logistic_growth(x)
            
        elif self.problem == 'emissivity_ss':
            
            responses = self.emissivity_ss(x)   

            
        elif self.problem == 'emissivity_inconel':
            
            responses = self.emissivity_inconel(x)  
            
        return responses
    

    
    


    
    
