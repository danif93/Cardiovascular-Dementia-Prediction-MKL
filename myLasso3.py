import numpy as np

def soft_thresholding(x, y):
    
    for i, xi in enumerate(x):
        if xi > y: x[i] = xi - y
        elif xi < - y: x[i] = xi + y
        else: x[i] = 0
    
    return x

def Lasso(M, a, c, tol=0.001, max_iter = 500, verbose = False):
    
    starting_max_iter = max_iter
    delta = -1
    eta = np.dot(np.linalg.pinv(M), a)
    eta /= np.linalg.norm(eta)
    
    while (delta > tol or delta < 0) and max_iter > 0:
    
        max_iter -= 1
        
        eta_new = soft_thresholding(eta + c * (np.dot(M, eta) - a), c) #it was a maximization problem, then + gradient
        norm =  np.linalg.norm(eta_new)
        if norm == 0:
            return eta
        
        eta_new /= norm
        
        delta = np.linalg.norm(eta_new - eta)
        eta = eta_new
            
        
    if max_iter == 0:
        print("Convergence not achieved in {} iterations".format(starting_max_iter))
        
    return eta 
    