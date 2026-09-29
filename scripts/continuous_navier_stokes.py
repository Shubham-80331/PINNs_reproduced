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

class PhysicsInformedNN_NavierStokes:
    def __init__(self, x, y, t, u, v, layers):
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        self.y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
        self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
        
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)
        
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))
        
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)
        
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        
    def net_NS(self, x, y, t):
        psi_and_p = self.dnn(torch.cat([x, y, t], dim=1))
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]
        
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        
        f_u = u_t + self.lambda_1 * (u*u_x + v*u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u*v_x + v*v_y) + p_y - self.lambda_2 * (v_xx + v_yy)
        
        return u, v, p, f_u, f_v
    
    def loss_func(self):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
        
        loss = torch.mean((self.u - u_pred)**2) + \
               torch.mean((self.v - v_pred)**2) + \
               torch.mean(f_u_pred**2) + \
               torch.mean(f_v_pred**2)
               
        return loss
        
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
    N_train = 5000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2] # 9 layers of 20
    
    data = scipy.io.loadmat('../data/cylinder_nektar_wake.mat')
    
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None]
    y = YY.flatten()[:,None]
    t = TT.flatten()[:,None]
    
    u = UU.flatten()[:,None]
    v = VV.flatten()[:,None]
    p = PP.flatten()[:,None]
    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    
    model = PhysicsInformedNN_NavierStokes(x_train, y_train, t_train, u_train, v_train, layers)
    
    start_time = time.time()
    model.train(epochs=100)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    print('lambda_1: %f' % (model.lambda_1.item()))
    print('lambda_2: %f' % (model.lambda_2.item()))
    
    error_lambda_1 = np.abs(model.lambda_1.item() - 1.0)*100
    error_lambda_2 = np.abs(model.lambda_2.item() - 0.01)/0.01 * 100
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))
    
    import matplotlib.pyplot as plt
    snap = 100
    x_star_snap = X_star[:,0:1]
    y_star_snap = X_star[:,1:2]
    t_star_snap = TT[:, snap:snap+1]
    p_exact = PP[:, snap:snap+1]
    
    model.dnn.eval()
    x_tensor = torch.tensor(x_star_snap, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_star_snap, dtype=torch.float32).to(device)
    t_tensor = torch.tensor(t_star_snap, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        psi_and_p = model.dnn(torch.cat([x_tensor, y_tensor, t_tensor], dim=1))
        p_pred = psi_and_p[:, 1:2].cpu().numpy()
        
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sc1 = ax[0].scatter(x_star_snap, y_star_snap, c=p_exact, cmap='rainbow')
    ax[0].set_title('Exact Pressure')
    fig.colorbar(sc1, ax=ax[0])
    
    sc2 = ax[1].scatter(x_star_snap, y_star_snap, c=p_pred, cmap='rainbow')
    ax[1].set_title('Predicted Pressure')
    fig.colorbar(sc2, ax=ax[1])
    
    plt.savefig('../results/continuous_navier_stokes.png')

if __name__ == "__main__":
    main()
