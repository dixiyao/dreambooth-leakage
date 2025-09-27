"""
An implementation of the loss function used in the DPGM method.

The code is referenced from the following repository:
https://github.com/dihjiang/DP-kernel
"""

import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal as mvn


class DP_loss(torch.nn.Module):
    """
    The RBF kerel for adding Gaussian noise to the kernel matrix.
    """

    def __init__(self, noise_multiplier, sigma_list=None):
        super(DP_loss, self).__init__()
        if sigma_list is None:
            sigma_list = [1, 2, 4, 8, 16]
        self.noise_multiplier = noise_multiplier
        self.sigma_list = sigma_list

    def forward(self, X, Y, reduction="mean"):
        """
        Compute Gaussian kernel between dataset X and Y
        :param X: N*d
        :param Y: M*d
        :return:
        """
        X = torch.flatten(X, start_dim=1)
        Y = torch.flatten(Y, start_dim=1)
        N = X.size(0)

        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()  # (N+M)*(N+M)

        K = 0.0
        for sigma in self.sigma_list:
            gamma = 1.0 / (2 * sigma**2)
            K += torch.exp(-gamma * exponent)

        K_XX = K[:N, :N]
        K_XY = K[:N, N:]
        K_YY = K[N:, N:]
        f_Dx = torch.mean(K_XX, dim=0)  # (N,)
        f_Dy = torch.mean(K_XY, dim=0)  # (M,)
        f_Dxy = torch.cat([f_Dx, f_Dy])  # size [N+M]

        # batch method
        coeff = math.sqrt(2 * len(self.sigma_list)) / N * self.noise_multiplier
        covariance = K * coeff
        # Eliminate error raised because that the values on diagonal lines are not equal due to the precision of float.
        covariance[1][0] = covariance[0][1]
        mvn_Dxy = mvn(torch.zeros_like(f_Dxy), covariance)
        f_Dxy_tilde = f_Dxy + mvn_Dxy.sample()
        f_Dx_tilde = f_Dxy_tilde[:N]  # [N]
        f_Dy_tilde = f_Dxy_tilde[N:]  # [M]
        del mvn_Dxy
        mmd_XX = torch.mean(f_Dx_tilde)
        mmd_XY = torch.mean(f_Dy_tilde)
        mmd_YY = torch.mean(K_YY)

        distance = mmd_XX - 2 * mmd_XY + mmd_YY
        return torch.pow(distance, 2)
