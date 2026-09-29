import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network import DNN
from src.plotting import plot_solution
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedNN_Discrete_AC:
    def __init__(self, x0, u0, x1, layers, dt, q, lb, ub):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(device)
        
        self.dt = dt
        self.q = q
        
        tmp = np.loadtxt('../data/Butcher_IRK%d.txt' % q, ndmin=1)
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        self.IRK_alpha = torch.tensor(self.IRK_weights[0:-1,:].T, dtype=torch.float32).to(device)
        self.IRK_beta = torch.tensor(self.IRK_weights[-1:,:].T, dtype=torch.float32).to(device)
        
        # The output of this network is q+1 dimensional: [u^{n+c_1}, ..., u^{n+c_q}, u^{n+1}]
        self.dnn = DNN(layers).to(device)
        
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        
    def net_U(self, x):
        X = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        U = self.dnn(X)
        return U
        
    def net_U0(self, x):
        U = self.net_U(x)
        # Network outputs (q+1) values
        U_c = U[:, :-1]
        U_1 = U[:, -1:]
        
        # Compute U_xx using a loop over each output component
        U_c_xx_list = []
        for i in range(self.q):
            u_c_i = U_c[:, i:i+1]
            u_c_i_x = torch.autograd.grad(u_c_i, x, grad_outputs=torch.ones_like(u_c_i), retain_graph=True, create_graph=True)[0]
            u_c_i_xx = torch.autograd.grad(u_c_i_x, x, grad_outputs=torch.ones_like(u_c_i_x), retain_graph=True, create_graph=True)[0]
            U_c_xx_list.append(u_c_i_xx)
            
        U_c_xx = torch.cat(U_c_xx_list, dim=1)
        
        # N(U_c) = -0.0001 U_c_xx + 5 U_c^3 - 5 U_c
        N_U_c = -0.0001 * U_c_xx + 5.0 * U_c**3 - 5.0 * U_c
        
        # U_0 = U_c + dt * N(U_c) * IRK_alpha
        # U_0_pred = U_1 + dt * N(U_c) * (IRK_alpha - IRK_beta)
        U_0 = U_c + self.dt * torch.matmul(N_U_c, self.IRK_alpha)
        U_0_pred = U_1 + self.dt * torch.matmul(N_U_c, (self.IRK_alpha - self.IRK_beta))
        
        return U_0, U_0_pred
    
    def net_U1(self, x):
        U = self.net_U(x)
        return U[:, -1:] # just the last column
        
    def loss_func(self):
        U_0, U_0_pred = self.net_U0(self.x0)
        
        # sum of squared errors
        loss_0 = torch.sum((self.u0 - U_0)**2) + torch.sum((self.u0 - U_0_pred)**2)
        
        # periodic boundary conditions
        x_lb = torch.tensor([[-1.0]], dtype=torch.float32, requires_grad=True).to(device)
        x_ub = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True).to(device)
        
        U_lb = self.net_U(x_lb)
        U_ub = self.net_U(x_ub)
        
        # derivative BC
        U_lb_x_list = []
        U_ub_x_list = []
        for i in range(self.q + 1):
            u_lb_i = U_lb[:, i:i+1]
            u_ub_i = U_ub[:, i:i+1]
            
            u_lb_i_x = torch.autograd.grad(u_lb_i, x_lb, retain_graph=True, create_graph=True)[0]
            u_ub_i_x = torch.autograd.grad(u_ub_i, x_ub, retain_graph=True, create_graph=True)[0]
            
            U_lb_x_list.append(u_lb_i_x)
            U_ub_x_list.append(u_ub_i_x)
            
        U_lb_x = torch.cat(U_lb_x_list, dim=1)
        U_ub_x = torch.cat(U_ub_x_list, dim=1)
        
        loss_b = torch.sum((U_lb - U_ub)**2) + torch.sum((U_lb_x - U_ub_x)**2)
        
        return loss_0 + loss_b
    
    def train(self, epochs):
        self.dnn.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4e}')
                
    def predict(self, x_star):
        self.dnn.eval()
        x = torch.tensor(x_star, dtype=torch.float32).to(device)
        with torch.no_grad():
            U1 = self.net_U1(x)
        return U1.cpu().numpy()

def main():
    q = 100
    layers = [1, 200, 200, 200, 200, q+1]
    dt = 0.8
    
    data = scipy.io.loadmat('../data/AC.mat')
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['uu']).T
    
    # We predict solution at t=0.9 using snapshot at t=0.1
    idx_t0 = 10  # t[10] = 0.1
    idx_t1 = 90  # t[90] = 0.9
    
    x0 = x
    u0 = Exact[idx_t0:idx_t0+1,:].T
    x1 = x
    
    N_n = 200
    idx_x = np.random.choice(x0.shape[0], N_n, replace=False)
    x0_train = x0[idx_x, :]
    u0_train = u0[idx_x, :]
    
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    model = PhysicsInformedNN_Discrete_AC(x0_train, u0_train, x1, layers, dt, q, lb, ub)
    
    start_time = time.time()
    model.train(epochs=100)
    print('Training time: %.4f' % (time.time() - start_time))
    
    u1_pred = model.predict(x1)
    
    u1_exact = Exact[idx_t1:idx_t1+1,:].T
    error_u1 = np.linalg.norm(u1_exact - u1_pred, 2) / np.linalg.norm(u1_exact, 2)
    print('Error u1: %e' % (error_u1))
    
    plt.figure()
    plt.plot(x1, u1_exact, 'b-', label='Exact')
    plt.plot(x1, u1_pred, 'r--', label='Predict')
    plt.legend()
    plt.title('Allen-Cahn Equation discrete time (t=0.9)')
    plt.savefig('../results/discrete_allen_cahn.png')
    
if __name__ == "__main__":
    main()
