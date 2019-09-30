import pandas as pd
import numpy as np
import scipy
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from config import Configuration
from utils import *
from model import *
class DR_Markowitz_Experiment(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.window = 12 # Try to evaluate annual return
        self.n = 109 # self.n: length of training periods 
        self.d = 20 # self.d: total number of assets
        
    def load_returns(self, seed, month):
        window = self.window
        train_path = self.data_path + 'Data/DataW/Data/seed' + str(seed) + '/NDDataPlay' + str(month) + '.csv'
        test_path = self.data_path + 'Data/DataW/Datatest/seed' + str(seed) + '/NDDataPlay' + str(month) + '.csv'
        prices_train_pd = pd.read_csv(train_path, delimiter=',', index_col='Unnamed: 0')
        prices_test_pd = pd.read_csv(test_path, delimiter=',', index_col='Unnamed: 0')
        n = prices_train_pd.shape[0]  # total number of instances of returns
        prices_train = prices_train_pd.astype(np.float32).values
        prices_test = prices_test_pd['x'].astype(np.float32).values
        returns_train = (prices_train[window:,:] - prices_train[:n-window,:])\
            / prices_train[:n-window, :] # compute the annual return
        returns_test = (prices_test - prices_train[-1,:]) \
            / prices_train[-1,:]
        self.n, self.d = returns_train.shape
        return returns_train, returns_test
        
    def load_implied_vol(self, month):
        window = self.window
        path = self.data_path + 'Data/VIX.csv'
        vol_data = pd.read_csv(path, delimiter=',')
        vol_open_data = vol_data['Open']
        vol_train = vol_open_data[(month+window):month+121]
        return vol_train
        
    def get_mahalanobis_matrix(self, seed, month, method = 'local'):
        vol_train = self.load_implied_vol(month)
        returns_train, returns_test = self.load_returns(seed, month)

        if method == 'local':
        	A = np.stack([(np.mean(vol_train)/vol) * np.eye(self.d) for vol in vol_train], axis = 2)
        elif method == 'covariance':
            A = np.stack([(np.mean(vol_train)/vol) * np.linalg.inv(np.cov(returns_train.T)) for vol in vol_train], axis = 2)
        else:
        	A = np.stack([np.eye(self.d) for vol in vol_train], axis = 2)
        return A
    
    def one_step_portfolio_return(self, seed, month, method = 'local'):
        returns_train, returns_test = self.load_returns(seed, month)
        A = self.get_mahalanobis_matrix(seed, month, method = method)
        model = DistRobustMarkowitz(X = returns_train, A = A)
        model.optimize(verbose = False, grad_method = 'analytic')
        one_step_return = returns_test.dot(model.beta)
        return one_step_return

    def portfolio_return(self, method = 'local'):
        seed_df = pd.read_csv(self.data_path+'Data/LinHistogramDataTangent.csv', delimiter=',')
        seeds = seed_df['Seed'][0:100]
        cores = cpu_count()
        p = Pool(cores)
        result = []
        # Testmonth is the number of month in the time period 2000-2017. In our case testmonth=12*17=204.
        testmonth = pd.read_csv(self.data_path+"Data/Nummonth.csv", delimiter=',')
        testmonth = testmonth['x'][0]
        pbar = tqdm(total=(len(seeds) * testmonth), leave = False)
        def update_pbar(*a):
            pbar.update()
        for idx,seed in enumerate(seeds):
            for month in range(1, testmonth + 1):
                result.append(p.apply_async(self.one_step_portfolio_return, args=(seed, month, method),
                                            callback = update_pbar))
        p.close()
        p.join()
        pbar.close()
        returns_mat = []
        for res in result:
            returns_mat.append(res.get())
        returns_mat = np.array(returns_mat).reshape(len(seeds), testmonth)
        return returns_mat

    def one_step_simulated_return(self, mean, cov, method = 'constant'):
        A = self.get_mahalanobis_matrix(1, 1, method = method)
        np.random.seed(int(time.time() * 1000) % (2 ** 31))
        exp_sqrt_cov = scipy.linalg.sqrtm(cov)
        exp_train = exp_sqrt_cov.dot(np.random.randn(20,109)).T + mean
        exp_test = exp_sqrt_cov.dot(np.random.randn(20)).T + mean
        model = DistRobustMarkowitz(X = exp_train, A = A)
        model.optimize(verbose = False, grad_method = 'analytic')
        one_step_return = exp_test.dot(model.beta)
        return one_step_return

    def simulated_portfolio_return(self, mean, cov, method = 'constant'):
        num_exp = 1000
        cores = cpu_count()
        p = Pool(cores)
        result = []
        pbar = tqdm(total=num_exp, leave = False)
        def update_pbar(*a):
            pbar.update()
        for i in range(num_exp):
            result.append(p.apply_async(self.one_step_simulated_return, args=(mean, cov, method),
                                        callback = update_pbar))
        p.close()
        p.join()
        pbar.close()
        returns_mat = []
        for res in result:
            returns_mat.append(res.get())
        return returns_mat