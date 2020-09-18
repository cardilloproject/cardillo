import numpy as np
from cardillo.math.numerical_derivative import Numerical_derivative

def verify_derivatives(mat):
    from numpy.random import rand
    from numpy import isclose

    #TODO: use normalized parameters
    #mat_temp = mat.__class__()
    
    rho = np.array([0.3,0.5])
    rho_s = np.array([[0.3,0.5],[0.1,0.8]])
    Gamma = np.array([0.4])
    theta_s = np.array([[0.1,0.6],[0.7,0.7]])

    W_rho_num_fun = lambda rho, rho_s, Gamma, theta_s: Numerical_derivative(lambda t, rho: mat.W(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)

    W_rho_s_num_fun = lambda rho, rho_s, Gamma, theta_s: Numerical_derivative(lambda rho_s: mat.W(rho, rho_s, Gamma, theta_s), order=1)._X(rho_s)

    W_Gamma_num_fun = lambda rho, rho_s, Gamma, theta_s: Numerical_derivative(lambda t, Gamma: mat.W(rho, rho_s, Gamma, theta_s), order=1)._x(0, Gamma)

    W_theta_s_num_fun = lambda rho, rho_s, Gamma, theta_s: Numerical_derivative(lambda theta_s: mat.W(rho, rho_s, Gamma, theta_s), order=1)._X(theta_s, eps=1e-5) # poorly conditioned due to lacking normalization

    first_deriv_dict = {
        "W_rho": W_rho_num_fun,
        "W_rho_s": W_rho_s_num_fun,
        "W_Gamma": W_Gamma_num_fun,
        "W_theta_s": W_theta_s_num_fun,
        }

    for key, element in first_deriv_dict.items():
        W_rho_an = eval("mat." + key + "(rho, rho_s, Gamma, theta_s)")
        W_rho_num = element(rho, rho_s, Gamma, theta_s)
        error = np.linalg.norm(W_rho_an - W_rho_num)
        if error > 1e-2:
            print(f"WARNING: error {key}: {error}")
        else:
            print(f"INFO: error {key}: {error}")
        

    second_deriv_dict = {
        "W_rho_rho": (lambda rho: mat.W_rho(rho, rho_s, Gamma, theta_s), rho),
        "W_rho_rho_s": (lambda rho_s: mat.W_rho(rho, rho_s, Gamma, theta_s), rho_s),
        "W_rho_theta_s": (lambda theta_s: mat.W_rho(rho, rho_s, Gamma, theta_s), theta_s),
        "W_rho_s_rho": (lambda rho: mat.W_rho_s(rho, rho_s, Gamma, theta_s), rho),
        "W_rho_s_rho_s": (lambda rho_s: mat.W_rho_s(rho, rho_s, Gamma, theta_s), rho_s),
        "W_Gamma_Gamma": (lambda Gamma: mat.W_Gamma(rho, rho_s, Gamma, theta_s), Gamma),
        "W_theta_s_rho": (lambda rho: mat.W_theta_s(rho, rho_s, Gamma, theta_s), rho),
        "W_theta_s_theta_s": (lambda theta_s: mat.W_theta_s(rho, rho_s, Gamma, theta_s), theta_s),
        }

    for key, element in second_deriv_dict.items():
        try:
            W_rho_rho_num = Numerical_derivative(element[0], order=1)._X(element[1])
            W_rho_rho_an = eval("mat." + key + "(rho, rho_s, Gamma, theta_s)")
            error = np.linalg.norm(W_rho_rho_num - W_rho_rho_an)
            if error > 1e-2:
                print(f"WARNING: error {key}: {error}")
            else:
                print(f"INFO: error {key}: {error}")
        except AttributeError:
            print(f"INFO: {key} not defined for selected material.")


class Maurin2019_linear():
    def __init__(self, K_rho, K_Gamma, K_Theta_s):
        self.K_rho = K_rho         # [Nm^-1]
        self.K_Gamma = K_Gamma     # [Nm^-1]
        self.K_Theta_s = K_Theta_s # [Nm]

    def W(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_rho * (rho[i] - 1)**2  
            W += 0.5 * self.K_Theta_s * theta_s[i,i]**2  # only s-s derivatives
        return W + 0.5 * self.K_Gamma * Gamma**2 

    def W_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * (rho - 1)

    def W_rho_s(self, rho, rho_s, Gamma, theta_s):
        return np.zeros((2, 2))

    def W_Gamma(self, rho, rho_s, Gamma, theta_s):
        return self.K_Gamma * Gamma
        
    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        return self.K_Theta_s * theta_s * np.eye(2) # only s-s derivatives

    def W_rho_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * np.eye(2)

    def W_Gamma_Gamma(self, rho, rho_s, Gamma, theta_s):
        return self.K_Gamma

    def W_theta_s_theta_s(self, rho, rho_s, Gamma, theta_s):
        W_theta_s_theta_s = np.zeros((2, 2, 2, 2))
        for i in range(2):
            W_theta_s_theta_s[i, i, i, i] = self.K_Theta_s
        return W_theta_s_theta_s

    # for post processing: axial, bending and shear strain energy density
    def W_axial(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_rho * (rho[i] - 1)**2  
        return W 

    def W_bending(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_Theta_s * theta_s[i,i]**2 
        return W

    def W_shear(self, rho, rho_s, Gamma, theta_s):
         return 0.5 * self.K_Gamma * Gamma**2 


class Maurin2019():
    def __init__(self, K_rho, K_Gamma, K_Theta_s, gamma):
        self.K_rho = K_rho         # [Nm^-1]
        self.K_Gamma = K_Gamma     # [Nm^-1]
        self.K_Theta_s = K_Theta_s # [Nm]
        self.gamma = gamma

    def W(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_rho * (rho[i] - 1)**2  
            W += 0.5 * self.K_Theta_s * theta_s[i,i]**2 
        return W + 0.5 * self.K_Gamma * np.abs(Gamma)**self.gamma

    # first derivatives
    def W_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * (rho - 1)

    def W_rho_s(self, rho, rho_s, Gamma, theta_s):
        return np.zeros((2, 2))

    def W_Gamma(self, rho, rho_s, Gamma, theta_s):
        # if np.isclose(Gamma, 0, atol=1e-02):
        #     gamma_new = (3 - self.gamma) / ((1e3 * Gamma)**2 + 1) + self.gamma
        #     return 0.5 * self.gamma * self.K_Gamma * np.sign(Gamma) * np.abs(Gamma)**(gamma_new - 1)
        return 0.5 * self.gamma * self.K_Gamma * np.sign(Gamma) * np.abs(Gamma)**(self.gamma - 1)
        
    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        return self.K_Theta_s * theta_s * np.eye(2) # only s-s derivatives

    # second derivatives
    def W_rho_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * np.eye(2)

    def W_Gamma_Gamma(self, rho, rho_s, Gamma, theta_s):
        # if np.isclose(Gamma, 0, atol=1e-02):
        #     gamma_new = (3 - self.gamma) / ((1e3 * Gamma)**2 + 1) + self.gamma
        #     return 0.5 * self.gamma * self.K_Gamma * (self.gamma - 1) * np.abs(Gamma)**(gamma_new - 2)
        return 0.5 * self.gamma * self.K_Gamma * (self.gamma - 1) * np.abs(Gamma)**(self.gamma - 2)

    def W_theta_s_theta_s(self, rho, rho_s, Gamma, theta_s):
        W_theta_s_theta_s = np.zeros((2, 2, 2, 2))
        for i in range(2):
            W_theta_s_theta_s[i, i, i, i] = self.K_Theta_s
        return W_theta_s_theta_s

    # for post processing: axial, bending and shear strain energy density
    def W_axial(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_rho * (rho[i] - 1)**2  
        return W 

    def W_bending(self, rho, rho_s, Gamma, theta_s):
        W = 0
        for i in range(2):
            W += 0.5 * self.K_Theta_s * theta_s[i,i]**2 
        return W

    def W_shear(self, rho, rho_s, Gamma, theta_s):
         return 0.5 * self.K_Gamma * np.abs(Gamma)**self.gamma


class Barchiesi2020():
    def __init__(self,gamma, K_F, K_E, K_S):
        self.gamma = gamma
        self.K_F = K_F   #[J]
        self.K_E = K_E  #[J]
        self.K_S = K_S    #[J]
        self.coga2 = np.cos(self.gamma)**2

    def W(self, rho, rho_s, Gamma, theta_s):
        # Barchiesi only uses diagonal elements of rho_s and theta_s
        den1 = rho**2 * self.coga2 * (self.K_E - 8*self.K_F*self.coga2) - self.K_E
        den2 = (1 - rho**2 * self.coga2 )* (8*self.K_F + rho**2 *(self.K_E - 8*self.K_F*self.coga2))

        W = np.sum(self.K_E * self.K_F *  rho**2 * self.coga2 / den2 * rho_s.diagonal()**2) 
        W += np.sum(self.K_E * self.K_F * (rho**2 * self.coga2 - 1) / den1 * theta_s.diagonal()**2) 
        W += np.sum(self.K_S * (np.arccos(1 - rho**2 * 2 * self.coga2) - np.pi + 2 * self.gamma)**2)
        return  W 

    def W_rho(self, rho, rho_s, Gamma, theta_s):
        den1 = rho**2 * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2) - self.K_E
        den2 = (1 - rho**2 * self.coga2 )* (8 * self.K_F + rho**2 *(self.K_E - 8 * self.K_F * self.coga2))

        return (theta_s.diagonal()**2 * self.K_E * self.K_F * ((2 * rho * self.coga2) / den1  - ((rho**2 * self.coga2 - 1) * (2 * rho * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2))) / den1**2) 
                + rho_s.diagonal()**2 * self.K_E * self.K_F * (2 * rho * self.coga2 / den2 - (rho**2 * self.coga2 * (2 * rho * (self.K_E - 16 * self.K_F * self.coga2) - 4 * rho**3 * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2))) / den2**2) 
                + 8 * self.K_S * (np.arccos(1- rho**2 * 2 * self.coga2) - np.pi + 2 * self.gamma) / np.sqrt(1-(1-rho**2 * 2* self.coga2)**2) * rho * self.coga2 
                )
    
    def W_rho_s(self, rho, rho_s, Gamma, theta_s):
        den2 = (1 - rho**2 * self.coga2) * (8 * self.K_F + rho**2 *(self.K_E - 8*self.K_F*self.coga2))
        return np.diag(rho**2 * self.coga2 / den2 * 2 * rho_s.diagonal()) * self.K_E * self.K_F

    def W_Gamma(self, rho, rho_s, Gamma, theta_s):
        return 0

    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        den1 = rho**2 * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2) - self.K_E
        return np.diag((rho**2 * self.coga2 -1 ) / den1 * 2 * theta_s.diagonal() * self.K_E * self.K_F )

    def W_rho_rho(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho: self.W_rho(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)

    def W_rho_rho_s(self, rho, rho_s, Gamma, theta_s):
        # return Numerical_derivative(lambda t, rho_s: self.W_rho(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho_s)
        den2 = (1 - rho**2 * self.coga2)* (8*self.K_F + rho**2 *(self.K_E - 8*self.K_F*self.coga2))
        W_rho_rho_s = np.zeros((2, 2, 2))
        W_rho_rho_s[np.diag_indices(2, ndim=3)] = 2 * rho_s.diagonal() * self.K_E * self.K_F * (2 * rho * self.coga2 / den2 - (rho**2 * self.coga2 * (2 * rho * (self.K_E - 16 * self.K_F * self.coga2) - 4 * rho**3 * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2))) / den2**2)
        return W_rho_rho_s

    def W_rho_theta_s(self, rho, rho_s, Gamma, theta_s):
        # return Numerical_derivative(lambda t, theta_s: self.W_rho(rho, rho_s, Gamma, theta_s), order=1)._x(0, theta_s)
        den1 = rho**2 * self.coga2 * (self.K_E - 8 * self.K_F*self.coga2) - self.K_E
        W_rho_theta_s = np.zeros((2, 2, 2))
        W_rho_theta_s[np.diag_indices(2, ndim=3)] = 2 * theta_s.diagonal() * self.K_E * self.K_F * ((2 * rho * self.coga2) / den1  - ((rho**2 * self.coga2 - 1) * (2 * rho * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2))) / den1**2)
        return W_rho_theta_s

    def W_rho_s_rho(self, rho, rho_s, Gamma, theta_s):
        # return Numerical_derivative(lambda t, rho: self.W_rho_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)
        den2 = (1 - rho**2 * self.coga2) * (8 * self.K_F + rho**2 * (self.K_E - 8 * self.K_F * self.coga2))
        den2_rho = (-2 * rho * self.coga2) * (8 * self.K_F + rho**2 * (self.K_E - 8 * self.K_F * self.coga2)) + (1 - rho**2 * self.coga2) * (2 * rho * (self.K_E - 8 * self.K_F * self.coga2))
        W_rho_s_rho = np.zeros((2, 2, 2))
        W_rho_s_rho[np.diag_indices(2, ndim=3)] = (2 * rho * self.coga2 / den2 - rho**2 * self.coga2 / den2**2 * den2_rho) * 2 * rho_s.diagonal() * self.K_E * self.K_F
        return W_rho_s_rho

    def W_rho_s_rho_s(self, rho, rho_s, Gamma, theta_s):
        # return Numerical_derivative(lambda t, rho_s: self.W_rho_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho_s)
        W_rho_s_rho_s = np.zeros((2, 2, 2, 2))
        den2 = (1 - rho**2 * self.coga2) * (8 * self.K_F + rho**2 * (self.K_E - 8 * self.K_F * self.coga2))
        W_rho_s_rho_s[np.diag_indices(2, ndim=4)] = (rho**2 * self.coga2 / den2 * 2) * self.K_E * self.K_F
        return W_rho_s_rho_s

    def W_Gamma_Gamma(self, rho, rho_s, Gamma, theta_s):
        return 0

    def W_theta_s_rho(self, rho, rho_s, Gamma, theta_s):
        # return Numerical_derivative(lambda t, rho: self.W_theta_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)
        den1 = rho**2 * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2) - self.K_E
        den1_rho = 2 * rho * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2)
        W_theta_s_rho = np.zeros((2, 2, 2))
        W_theta_s_rho[np.diag_indices(2, ndim=3)] = ((2 * rho * self.coga2) / den1 - (rho**2 * self.coga2 -1 ) / den1**2 * den1_rho) * 2 * theta_s.diagonal() * self.K_E * self.K_F 
        return W_theta_s_rho

    def W_theta_s_theta_s(self, rho, rho_s, Gamma, theta_s):
        # return Numerical_derivative(lambda t, rho: self.W_theta_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, theta_s)
        W_theta_s_theta_s = np.zeros((2, 2, 2, 2))
        den1 = rho**2 * self.coga2 * (self.K_E - 8 * self.K_F * self.coga2) - self.K_E
        W_theta_s_theta_s[np.diag_indices(2, ndim=4)] = (rho**2 * self.coga2 -1) / den1 * 2 * self.K_E * self.K_F 
        return W_theta_s_theta_s

