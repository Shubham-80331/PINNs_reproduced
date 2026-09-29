import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network import DNN
from src.utils import latin_hypercube_sample
from src.plotting import plot_solution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedNN_Schrodinger:
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.t0 = torch.zeros_like(self.x0).to(device)
        
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.v0 = torch.tensor(v0, dtype=torch.float32).to(device)
        
        self.tb = torch.tensor(tb, dtype=torch.float32, requires_grad=True).to(device)
        self.x_lb = torch.full_like(self.tb, lb[0], requires_grad=True).to(device)
        self.x_ub = torch.full_like(self.tb, ub[0], requires_grad=True).to(device)
        
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        
        # 2 outputs for u and v
        self.dnn = DNN(layers).to(device)
        
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        
    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.dnn(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v
    
    def net_f(self, x, t):
        u, v = self.net_uv(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        
        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
        
        return f_u, f_v
    
    def loss_func(self):
        u0_pred, v0_pred = self.net_uv(self.x0, self.t0)
        
        loss_0 = torch.mean((self.u0 - u0_pred)**2) + torch.mean((self.v0 - v0_pred)**2)
        
        u_lb, v_lb = self.net_uv(self.x_lb, self.tb)
        u_ub, v_ub = self.net_uv(self.x_ub, self.tb)
        
        u_x_lb = torch.autograd.grad(u_lb, self.x_lb, grad_outputs=torch.ones_like(u_lb), retain_graph=True, create_graph=True)[0]
        u_x_ub = torch.autograd.grad(u_ub, self.x_ub, grad_outputs=torch.ones_like(u_ub), retain_graph=True, create_graph=True)[0]
        v_x_lb = torch.autograd.grad(v_lb, self.x_lb, grad_outputs=torch.ones_like(v_lb), retain_graph=True, create_graph=True)[0]
        v_x_ub = torch.autograd.grad(v_ub, self.x_ub, grad_outputs=torch.ones_like(v_ub), retain_graph=True, create_graph=True)[0]
        
        loss_b = torch.mean((u_lb - u_ub)**2) + torch.mean((v_lb - v_ub)**2) + \
                 torch.mean((u_x_lb - u_x_ub)**2) + torch.mean((v_x_lb - v_x_ub)**2)
                 
        f_u_pred, f_v_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_u_pred**2) + torch.mean(f_v_pred**2)
        
        return loss_0 + loss_b + loss_f
        
    def train(self, epochs):
        self.dnn.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4e}')
                
    def predict(self, X_star):
        self.dnn.eval()
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(device)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            u, v = self.net_uv(x, t)
            h = torch.sqrt(u**2 + v**2)
        return h.cpu().numpy()

def main():
    N_0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2] # 4 hidden layers of 100
    
    data = scipy.io.loadmat('../data/NLS.mat')
    
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_h = np.sqrt(np.real(Exact)**2 + np.imag(Exact)**2)
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    h_star = Exact_h.T.flatten()[:,None]
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    # IC
    idx_x = np.random.choice(x.shape[0], N_0, replace=False)
    x0 = x[idx_x,:]
    u0 = np.real(Exact[idx_x,0:1])
    v0 = np.imag(Exact[idx_x,0:1])
    
    # BC
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    # Collocation points
    X_f = latin_hypercube_sample(N_f, list(zip(lb, ub)))
    
    model = PhysicsInformedNN_Schrodinger(x0, u0, v0, tb, X_f, layers, lb, ub)
    
    start_time = time.time()
    model.train(epochs=1000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    h_pred = model.predict(X_star)
    
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error h: %e' % (error_h))
    
    H_pred = scipy.interpolate.griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
    fig = plot_solution(X, T, Exact_h.T, H_pred, title="Schrodinger Equation |h(t,x)|")
    fig.savefig('../results/continuous_schrodinger.png')

if __name__ == "__main__":
    main()
