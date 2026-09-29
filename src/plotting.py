import matplotlib.pyplot as plt
import numpy as np

def plot_solution(X, T, Exact, u_pred, title=""):
    fig = plt.figure(figsize=(10, 5))
    
    # Exact
    ax1 = fig.add_subplot(121)
    h1 = ax1.imshow(Exact.T, interpolation='nearest', cmap='rainbow', 
                  extent=[T.min(), T.max(), X.min(), X.max()], 
                  origin='lower', aspect='auto')
    plt.colorbar(h1, ax=ax1)
    ax1.set_title('Exact ' + title)
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    
    # Prediction
    ax2 = fig.add_subplot(122)
    h2 = ax2.imshow(u_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[T.min(), T.max(), X.min(), X.max()], 
                  origin='lower', aspect='auto')
    plt.colorbar(h2, ax=ax2)
    ax2.set_title('Predict ' + title)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    
    plt.tight_layout()
    return fig
