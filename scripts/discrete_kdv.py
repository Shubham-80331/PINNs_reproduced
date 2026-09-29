import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network import DNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedNN_Discrete_KdV:
    def __init__(self, x0, u0, x1, u1, layers, dt, q, lb, ub):
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(device)
        
        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.u1 = torch.tensor(u1, dtype=torch.float32).to(device)
        
        self.dt = dt
        self.q = q
        
        # Load IRK weights
        tmp = np.loadtxt('../data/Butcher_IRK%d.txt' % q, ndmin=1)
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        self.IRK_alpha = torch.tensor(self.IRK_weights[0:-1,:].T, dtype=torch.float32).to(device)
        self.IRK_beta = torch.tensor(self.IRK_weights[-1:,:].T, dtype=torch.float32).to(device)
        
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))
        
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)
        
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        
    def net_U(self, x):
        X = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        U = self.dnn(X)
        return U
        
    def net_U0(self, x):
        U = self.net_U(x)
        U_c = U[:, :-1]
        U_1 = U[:, -1:]
        
        U_c_x_list = []
        U_c_xxx_list = []
        for i in range(self.q):
            u_c_i = U_c[:, i:i+1]
            u_c_i_x = torch.autograd.grad(u_c_i, x, grad_outputs=torch.ones_like(u_c_i), retain_graph=True, create_graph=True)[0]
            u_c_i_xx = torch.autograd.grad(u_c_i_x, x, grad_outputs=torch.ones_like(u_c_i_x), retain_graph=True, create_graph=True)[0]
            u_c_i_xxx = torch.autograd.grad(u_c_i_xx, x, grad_outputs=torch.ones_like(u_c_i_xx), retain_graph=True, create_graph=True)[0]
            
            U_c_x_list.append(u_c_i_x)
            U_c_xxx_list.append(u_c_i_xxx)
            
        U_c_x = torch.cat(U_c_x_list, dim=1)
        U_c_xxx = torch.cat(U_c_xxx_list, dim=1)
        
        # N(U_c) = lambda_1 * U_c * U_c_x + lambda_2 * U_c_xxx
        N_U_c = self.lambda_1 * U_c * U_c_x - self.lambda_2 * U_c_xxx
        
        U_0 = U_c + self.dt * torch.matmul(N_U_c, self.IRK_alpha)
        U_0_pred = U_1 + self.dt * torch.matmul(N_U_c, (self.IRK_alpha - self.IRK_beta))
        
        return U_0, U_0_pred, U_1
        
    def loss_func(self):
        U_0, U_0_pred, U_1_pred = self.net_U0(self.x0)
        
        loss_0 = torch.mean((self.u0 - U_0)**2) + torch.mean((self.u0 - U_0_pred)**2)
        
        # for x1, compute U1 directly
        U_1_pred_x1 = self.net_U(self.x1)[:, -1:]
        loss_1 = torch.mean((self.u1 - U_1_pred_x1)**2)
        
        return loss_0 + loss_1
    
    def train(self, epochs):
        self.dnn.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4e}, l1: {self.lambda_1.item():.4f}, l2: {self.lambda_2.item():.5f}')

def main():
    q = 50
    layers = [1, 50, 50, 50, 50, q+1]
    
    data = scipy.io.loadmat('../data/KdV.mat')
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['uu']).T
    
    idx_t0 = 40
    idx_t1 = 160
    dt = t[idx_t1] - t[idx_t0]
    dt = dt.item()
    
    # Training Data
    N0 = 199
    N1 = 201
    
    idx_x0 = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x0,:]
    u0 = Exact[idx_t0:idx_t0+1,idx_x0].T
    
    idx_x1 = np.random.choice(x.shape[0], N1, replace=False)
    x1 = x[idx_x1,:]
    u1 = Exact[idx_t1:idx_t1+1,idx_x1].T
    
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    model = PhysicsInformedNN_Discrete_KdV(x0, u0, x1, u1, layers, dt, q, lb, ub)
    
    start_time = time.time()
    model.train(epochs=100)
    print('Training time: %.4f' % (time.time() - start_time))
    
    print('lambda_1: %f' % (model.lambda_1.item()))
    print('lambda_2: %f' % (model.lambda_2.item()))
    
    error_lambda_1 = np.abs(model.lambda_1.item() - 1.0)*100
    error_lambda_2 = np.abs(model.lambda_2.item() - 0.0025)/0.0025 * 100
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))
    
    import matplotlib.pyplot as plt
    u1_pred = model.net_U(torch.tensor(x1, dtype=torch.float32).to(device))[:, -1:].cpu().detach().numpy()
    
    plt.figure()
    plt.plot(x1, u1, 'b-', label='Exact')
    plt.plot(x1, u1_pred, 'r--', label='Predict')
    plt.legend()
    plt.title('KdV Equation (t = 0.8)')
    plt.savefig('../results/discrete_kdv.png')
    
if __name__ == "__main__":
    main()
