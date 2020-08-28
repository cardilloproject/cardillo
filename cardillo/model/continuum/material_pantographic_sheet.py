import numpy as np
from cardillo.math.numerical_derivative import Numerical_derivative

# def verify_derivatives(W_func, dW_func):
#     eps = np.array([0.1, 0.2])
#     eps_al = np.array([2, 3])
#     theta = np.array([0.7, 1.2])
#     theta_al = np.array([0.2, 0.3])
#     Eps0 = np.vstack((eps, eps_al, theta, theta_al))

#     dW_num = Numerical_derivative(lambda t, Eps: np.sum(W_func(*Eps)), order=1)._x(0, Eps0)
#     dW_ana = dW_func(*Eps0)

#     error = np.linalg.norm(dW_num - dW_ana)
#     print(f'error: {error}')

class Maurin2019_linear():
    def __init__(self, K_rho, K_Gamma, K_Theta_s):
        self.K_rho = K_rho         # [Nm^-1]
        self.K_Gamma = K_Gamma     # [Nm^-1]
        self.K_Theta_s = K_Theta_s # [Nm]

    def W(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_rho * (rho[i] - 1)**2  
            W += 0.5 * self.K_Theta_s * theta_s[i]**2 
        return W + 0.5 * self.K_Gamma * Gamma**2

    def W_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * (rho - 1)

    def W_rho_s(self, rho, rho_s, Gamma, theta_s):
        return np.zeros((2, 2))

    def W_Gamma(self, rho, rho_s, Gamma, theta_s):
        return self.K_Gamma * Gamma
        
    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        return self.K_Theta_s * theta_s 

# class Maurin2019():
#     # def __init__(self, gamma, K_eps, K_kap, K_theta):
#     #     self.gamma = gamma
#     def __init__(self, K_rho, K_Gamma, K_Theta_s):
#         self.K_rho = K_rho         # [Nm^-1]
#         self.K_Gamma = K_Gamma     # [Nm^-1]
#         self.K_Theta_s = K_Theta_s # [Nm]

#     def W(self, rho, rho_s, Gamma, theta_s):
#         W = 0
#         for i in range(2):
#             W += 0.5 * self.K_rho * (rho[i] - 1)**2  
#             W += 0.5 * self.K_Theta_s * theta_s[i]**2 
        
#         # # TODO: 0.5 * |Gamma|^(gamma)?
#         # W += self.K_theta * np.abs(Gamma)**(self.gamma)
#         W += 0.5 * self.K_Gamma * Gamma**2

#         return W

#     def W_rho(self, rho, rho_s, Gamma, theta_s):
#         return self.K_rho * (rho - 1)

#     def W_rho_s(self, rho, rho_s, Gamma, theta_s):
#         return np.zeros((2, 2))

#     def W_Gamma(self, rho, rho_s, Gamma, theta_s):
#         # return self.gamma * self.K_theta * np.abs(Gamma)**(self.gamma - 1)
#         return self.K_Gamma * Gamma
        
#     def W_Gamma_s(self, rho, rho_s, Gamma, theta_s):
#         return self.K_Theta_s * theta_s