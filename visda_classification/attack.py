import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import to_var, sample_unit_vec

# --- White-box attacks ---

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon

    def loss_fn(self, out1, out2):
        # print(out2.data.type())
        if out2.data.type()=='torch.cuda.LongTensor' or out2.data.type()=='torch.LongTensor':
            return nn.CrossEntropyLoss()(F.softmax(out1, dim=1), out2)
        else:
            return - torch.mean(F.softmax(out2, dim=1) * torch.log(F.softmax(out1, dim=1) + 1e-6))

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.from_numpy(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X

class IFGSMAttack(object):
    def __init__(self, model=None, epsilon=None, momentum=None, num_k=1):
        """
        Iterative fast gradient sign method with momentum
        """
        self.model = model
        self.epsilon = epsilon
        self.momentum = momentum
        self.num_k = num_k

    def loss_fn(self, out1, out2):
        # print(out2.data.type())
        if out2.data.type()=='torch.cuda.LongTensor' or out2.data.type()=='torch.LongTensor':
            return nn.CrossEntropyLoss()(F.softmax(out1, dim=1), out2)
        else:
            return - torch.mean(F.softmax(out2, dim=1) * torch.log(F.softmax(out1, dim=1) + 1e-6))

    def perturb(self, X_nat, y, epsilons=None, momentum=None, num_k=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons
        if momentum is not None:
            self.momentum = momentum
        if num_k is not None:
            self.num_k = num_k

        X = np.copy(X_nat)
        S = np.zeros_like(X)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.from_numpy(y))

        for i in range(self.num_k):
            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            # grad_sign = X_var.grad.data.cpu().sign().numpy()
            grad = X_var.grad.data
            grad_mean = torch.mean(torch.abs(grad))
            grad_norm = grad.cpu().numpy() / grad_mean
            if i == 0:
                S = grad_norm
            else:
                S = S*self.momentum + (1-self.momentum)*grad_norm

            X += self.epsilon * S
            X = np.clip(X, 0, 1)
            X_var = to_var(torch.from_numpy(X), requires_grad=True)

        return X

class VATAttack(object):
    def __init__(self, model=None, epsilon=None, zeta=None, num_k=1):
        """
        Fast approximation method in virtual adversarial training
        :param model: nn.Module
        :param epsilon: float
        :param zeta: float
        :param num_k: int, number of iterations
        """
        self.model = model
        self.epsilon = epsilon
        self.zeta = zeta
        self.num_k = num_k

    def loss_fn(self, out1, out2):
        if out2.data.type()=='torch.cuda.LongTensor' or out2.data.type()=='torch.LongTensor':
            return nn.KLDivLoss()(F.softmax(out1, dim=1), out2)
        else:
            return torch.mean(F.softmax(out2, dim=1) *
                              (torch.log(F.softmax(out2, dim=1) + 1e-6) - torch.log(F.softmax(out1, dim=1) + 1e-6)))

    def perturb(self, X_nat, epsilons=None, zetas=None):
        """
        Given examples (X_nat), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons
        if zetas is not None:
            self.zeta = zetas

        X = np.copy(X_nat)
        d = sample_unit_vec(X.shape[1:], X.shape[0])

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        d_var = to_var(d, requires_grad=True)

        for i in range(self.num_k):
            r_var = self.zeta * d_var
            ref = self.model(X_var)
            pert = self.model(X_var + r_var)
            loss = self.loss_fn(pert, ref)
            loss.backward()
            d_var = to_var(d_var.grad.data, requires_grad=True)

        X += self.epsilon * (self.zeta * d_var).data.cpu().sign().numpy()
        X = np.clip(X, 0, 1)
        return X
