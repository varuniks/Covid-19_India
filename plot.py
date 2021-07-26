import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_state_prediction(states_list, true, pred, mode):
    print(true.shape)
    print(pred.shape)

    s = 0 
    for m in range(4):
        for n in range(9):
            ax = plt.subplot(3, 3, n + 1)
            ax.plot(true[:,0,s], label='Truth')
            ax.plot(pred[:,0,s], label='pred')
            ax.title.set_text(states_list[s])
            s = s+1
        plt.suptitle(mode)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./plots/'+mode+str(m)+'.png')
        #plt.show()
        plt.close()
        

