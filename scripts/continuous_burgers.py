import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network import DNN
from src.utils import latin_hypercube_sample, numpy_to_tensor
from src.plotting import plot_solution

# CUDA support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        self.dnn = DNN(layers).to(device)
        
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        
    def net_u(self, x, t):
        # Scale inputs to [-1, 1]
        X = torch.cat([x, t], dim=1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.dnn(X)
    
    def net_f(self, x, t):
        u = self.net_u(x, t)
        
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True
        )[0]
        
        f = u_t + u * u_x - (0.01 / np.pi) * u_xx
        return f
    
    def loss_func(self):
        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        return loss_u + loss_f
    
    def train(self, epochs):
        self.dnn.train()
        for epoch in range(epochs):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_Adam.step()
            
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4e}')
                
    def predict(self, X_star):
        self.dnn.eval()
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(device)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            u = self.net_u(x, t)
        return u.cpu().numpy()

def main():
    nu = 0.01/np.pi
    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    data = scipy.io.loadmat('../data/burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              
    
    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*latin_hypercube_sample(N_f, [[0, 1], [0, 1]]) # LHS in [0,1]^2 then scaled
    # Actually our LHS helper returns scaled directly if we pass bounds
    X_f_train = latin_hypercube_sample(N_f, list(zip(lb, ub)))
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
    
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
    
    start_time = time.time()
    # Train for a few epochs for testing (1000)
    model.train(epochs=1000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred = model.predict(X_star)
    
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     
    
    U_pred = scipy.interpolate.griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    fig = plot_solution(X, T, Exact, U_pred, title="Burgers Equation")
    fig.savefig('../results/continuous_burgers.png')

if __name__ == "__main__":
    main()
