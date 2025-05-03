import torch
import torch.nn.functional as F
import numpy as np
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')
    

class MAB():
    def __init__(self, n_models, rho=1, eps=0.5, momentum=0.01, a=1, b=1, update_rule="EMA", policy='UCB'):
        self.n_models = n_models
        self.rho = rho
        self.eps = eps
        self.Q0 = np.ones(n_models)*np.inf
        self.step_n = 0
        self.step_arm = np.zeros(n_models)
        self.AM_reward = np.zeros(n_models)
        self.policy = policy
        self.momentum = momentum
        self.update_rule = update_rule
        
        # Beta parameters for the prior in Thompson Sampling
        self.a = a
        self.b = b

    def update(self, index, value):
        self.step_n += 1
        self.step_arm[index] += 1
        if self.update_rule == 'Standard':
            # Standard update rule
            self.AM_reward[index] = ((self.step_arm[index] - 1) / float(self.step_arm[index]) 
            * self.AM_reward[index] + (1 / float(self.step_arm[index])) * value)
        elif self.update_rule == 'EMA':
            # Momentum update rule EMA
            self.AM_reward[index] = (1 - self.momentum) * self.AM_reward[index] + self.momentum * value
        else:
            raise ValueError('Invalid update rule')
        
    def get_stats(self):
        return {'step_n': self.step_n, 'step_arm': self.step_arm.tolist(), 'AM_reward': self.AM_reward.tolist()}

    def __call__(self, k=1):
        if self.policy == 'UCB':
            if len(np.where(self.step_arm == 0)[0]) > k:
                return np.random.choice(np.where(self.step_arm == 0)[0], k, replace=False)
            ucb_values = np.zeros(self.n_models)
            for i in range(self.n_models):
                if self.step_arm[i] == 0:
                    ucb_values[i] = self.Q0[i]
                else:
                    ucb_values[i] = np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[i])
            Q = self.AM_reward + ucb_values
            sorted_index = np.argsort(Q)[::-1]
            return sorted_index[:k]
        elif self.policy == 'EpsGreedy':
            if np.random.rand() < self.eps:
                return np.random.choice(np.arange(self.n_models), k, replace=False)
            else:
                return np.argsort(self.AM_reward)[::-1][:k]
        elif self.policy == 'Random':
            return np.random.choice(np.arange(self.n_models), k, replace=False)
        elif self.policy == 'ThompsonSampling':
            samples = np.random.beta(self.a + self.AM_reward, self.b + self.step_arm - self.AM_reward)
            return np.argsort(samples)[::-1][:k]
        raise ValueError('Invalid policy')