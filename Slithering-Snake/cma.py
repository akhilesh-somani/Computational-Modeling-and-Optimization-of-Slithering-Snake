# Akhilesh

import numpy as np
import pylab as plt
import math as mth
import matplotlib.pyplot as plt
from functools import partial
import copy
from multiprocessing import Pool
import time

import pickle

class CMAES:
    
    def __init__(self, initial_mean, sigma, popsize, iterations, verbose=False, **kwargs):
        # Add multiprocessing pool
        self.n_pool = kwargs.get('n_pool', 1)
        if self.n_pool == 1:
            pass
        else:
            self.pool = Pool(processes=self.n_pool)

        # Setup
        self.verbose = verbose
        
        # Things that evolve : centroid, sigma, paths etc.
        self.centroid = np.asarray(initial_mean).copy()
        self.sigma = sigma

        # pc is the path taken by the covariance matrix
        self.pc = np.zeros((initial_mean.shape[0],))

        # ps is the path taken by sigma / step-size updates
        self.ps = np.zeros((initial_mean.shape[0],))
        
        self.C = np.identity(initial_mean.shape[0])
        self.B = np.identity(initial_mean.shape[0])
        self.diagD = np.ones(initial_mean.shape[0])
        
        # Utility variables
        self.dim = initial_mean.shape[0]
        
        # Population size etc. 
        self.popsize = popsize
        
        self.mu=int(np.floor(self.popsize/2.))
        
        # Update weights
        self.weights = np.arange(self.mu,0.0,-1.0)
        self.weights /= np.sum(self.weights)

        # Expectation of a normal distribution
        self.chiN = np.sqrt(self.dim) * (1.0 - 0.25 / self.dim + 1.0/(21.0 * self.dim**2))  # Expected value
        self.mueff = 1/np.sum([i**2 for i in self.weights])
        self.generations = 0
 
        ######### Sigma adaptation
        # cs is short for c_sigma
        self.cs = (self.mueff+2)/(self.dim+self.mueff+5)
        #print("self.cs is",self.cs)
        
        # ds is short for d_sigma
        self.ds = 1+2*np.amax([0,np.sqrt((self.mueff-1)/(self.dim+1))-1])+self.cs
        #print("self.ds is",self.ds)
        
        ######### Covariance adaptation
        self.cc = (4+self.mueff/self.dim)/(self.dim+4+2*self.mueff/self.dim)
        self.ccov = 0.15
        
        self.total_iter = iterations
        
        # Collect useful statistics 
        # self.stats_centroids = []
        # self.stats_new_centroids = []
        # self.stats_covs = []
        # self.stats_new_covs = []
        # self.stats_offspring = []
        # self.stats_offspring_weights = []
        # self.stats_ps = []

    def save_state(self, path):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        pickle_out = open(path, "wb")
        pickle.dump(self_dict, pickle_out)
        pickle_out.close()

    def load_state(self, path):
        pickle_in = open(path, "rb")
        self_dict = pickle.load(pickle_in)
        for k in self_dict.keys():
            setattr(self, k, self_dict[k])
      
    def run(self, problem):  
        best_output=[]
        
        # Keep track of how many iterations to perform
        while np.linalg.cond(np.diag(self.diagD))<10**14 and self.generations < self.total_iter:
            # Sample the population here!
            stime = time.time()
            z_i = np.random.multivariate_normal(np.zeros(self.dim,), self.C, self.popsize)
            x_i = [self.centroid+np.array(i)*self.sigma for i in z_i]
            # Pass the population to update, which computes all new parameters
            # while sorting the population
            
            fitness, best_pop = self.update(problem, x_i, z_i) # fitness here represents best fitness for that iteration

            self.save_state(f'run_gen{self.generations}.pkl')
            if self.verbose:
                print('fitness: ', fitness, ' best: ', best_pop, ' time: ', time.time()-stime)
            
            best_output.append(fitness)
            
            # increment generation counter
            self.generations += 1
    
            
        # returns the best individual at the last generation (ie the best solution at the end of simulation)
        # best_output contains the fitness values tracked over all the iterations
        return best_pop, best_output

    def update(self, problem, x_i, z_i):
        # -- store current state of the algorithm
        # self.stats_centroids.append(copy.deepcopy(self.centroid))
        # self.stats_covs.append(copy.deepcopy(self.C))
        
        # Sort the population here and work with only the sorted population     
        x_i=np.array(x_i)

        if self.n_pool == 1:
            f_i=[problem(x) for x in x_i]
        else:
            f_i = list(self.pool.map(problem, x_i))
        idx=np.argsort(f_i)
        f_i=np.sort(f_i)
        z_f=[z_i[idx[i]] for i in range(self.mu)]
        
        best_output = x_i[idx[0]]
        
        # -- store sorted offspring
        # self.stats_offspring.append(copy.deepcopy(z_f))
        
        # Store old centroid in-case
        old_centroid = self.centroid
        # Update centroid to self.centroid here
        z_w=[self.weights[i]*np.array(z_f[i]) for i in range(self.mu)]
        z_w=sum(z_w, 0).tolist()
                    
        self.centroid = old_centroid+np.array(z_w)*self.sigma
        
        # -- store new centroid
        # self.stats_new_centroids.append(copy.deepcopy(self.centroid))
        
        # Cumulation : update evolution path 
        self.ps = (1-self.cs)*np.array(self.ps)+np.sqrt(1-(1-self.cs)**2)*np.sqrt(self.mueff)*(self.B*(np.linalg.inv(np.diag(self.diagD)))*np.linalg.inv(self.B))@np.array(z_w)
               
        # -- store new evol path
        # self.stats_ps.append(copy.deepcopy(self.ps))
        
        # Cumulation : update evolution path for centroid
        self.pc = (1-self.cc)*np.array(self.pc)+np.sqrt(1-(1-self.cc)**2)*np.sqrt(self.mueff)*np.array(z_w)
                
        # Update covariance matrix
    
        Z=[self.weights[i]*np.outer(z_f[i],z_f[i]) for i in range(self.mu)]
        Z=sum(Z, 0).tolist()
                
        self.C = (1-self.ccov)*self.C + (self.ccov/np.sqrt(self.mueff))*np.outer(self.pc,(self.pc).T)+self.ccov*(1-1/np.sqrt(self.mueff))*np.array(Z)
        
        # -- store new covs
        # self.stats_new_covs.append(copy.deepcopy(self.C))
        
        # Update new sigma in-place, can be done before too
        self.sigma *= np.exp(self.cs/self.ds*(np.linalg.norm(self.ps)/self.chiN-1))       
        
        # Get the eigen decomposition for the covariance matrix to calculate inverse
        diagD_squared, self.B = np.linalg.eigh(self.C)
        self.diagD = np.sqrt((diagD_squared))        
        
        return f_i[0], best_output
