import numpy as np
from cardillo.math.numerical_derivative import Numerical_derivative

def verify_derivatives(mat):
    from numpy.random import rand
    from numpy import isclose

    #TODO: use normalized parameters
    #mat_temp = mat.__class__()
    
    rho = rand(2)
    rho_s = rand(2,2)
    Gamma = rand(1)
    theta_s = rand(2,2)

    W_rho_num = Numerical_derivative(lambda t, rho: mat.W(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho).ravel()
    W_rho_an = mat.W_rho(rho, rho_s, Gamma, theta_s).ravel()

    W_rho_s_num = Numerical_derivative(lambda rho_s: mat.W(rho, rho_s, Gamma, theta_s), order=1)._X(rho_s).ravel()
    W_rho_s_an = mat.W_rho_s(rho, rho_s, Gamma, theta_s).ravel()

    W_Gamma_num = Numerical_derivative(lambda t, Gamma: mat.W(rho, rho_s, Gamma, theta_s), order=1)._x(0, Gamma).ravel()
    W_Gamma_an = mat.W_Gamma(rho, rho_s, Gamma, theta_s).ravel()

    W_theta_s_num = Numerical_derivative(lambda theta_s: mat.W(rho, rho_s, Gamma, theta_s), order=1)._X(theta_s, eps=1e-5).ravel() # poorly conditioned due to lacking normalization
    W_theta_s_an = mat.W_theta_s(rho, rho_s, Gamma, theta_s).ravel()

    if (not np.isclose(W_rho_num, W_rho_an, rtol=1e-04).all() or 
        not np.isclose(W_rho_s_num, W_rho_s_an, rtol=1e-04).all() or 
        not np.isclose(W_Gamma_num, W_Gamma_an, rtol=1e-04).all() or  
        not np.isclose(W_theta_s_num, W_theta_s_an, rtol=1e-02).all() #TODO: implement stronger check than rtol=1e-02
    ):
        raise ValueError('The analytic derivatives in the material do not match the numerical derivatives')

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
        return 0.5 * self.gamma * self.K_Gamma * np.sign(Gamma) * np.abs(Gamma)**(self.gamma - 1)
        
    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        return self.K_Theta_s * theta_s * np.eye(2) # only s-s derivatives

    # second derivatives
    def W_rho_rho(self, rho, rho_s, Gamma, theta_s):
        return self.K_rho * np.eye(2)

    def W_Gamma_Gamma(self, rho, rho_s, Gamma, theta_s):
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
        self.K_F = 0.9   #[J]
        self.K_E = 0.33  #[J]
        self.K_S = 34    #[J]
        self.coga2 = np.cos(self.gamma)**2

    def W(self, rho, rho_s, Gamma, theta_s):
        den1 = rho**2 * self.coga2 * (self.K_E - 8*self.K_F*self.coga2) - self.K_E
        den2 = (1 - rho**2 * self.coga2 )* (8*self.K_F + rho**2 *(self.K_E - 8*self.K_F*self.coga2))

        W = np.sum(self.K_E * self.K_F *  rho**2 * self.coga2 / den2 * rho_s**2  ) 
        W += np.sum(self.K_E * self.K_F * (rho**2 * self.coga2 - 1)/den1 * theta_s**2 ) 
        W += np.sum(   self.K_S * (np.arccos(1 - rho**2 * 2 * self.coga2) - np.pi + 2*self.gamma)**2  )
        return  W 

    def W_rho(self, rho, rho_s, Gamma, theta_s):
        den1 = (rho**2 * self.coga2 * (self.K_E - 8*self.K_F*self.coga2) - self.K_E)
        den2 = (1 - rho**2 * self.coga2 )* (8*self.K_F + rho**2 *(self.K_E - 8*self.K_F*self.coga2))

        return (theta_s**2 * self.K_E*self.K_F * ((2*rho*self.coga2 )/den1  - ((rho**2 * self.coga2-1)*(2*rho*self.coga2*(self.K_E - 8*self.K_F*self.coga2)))/den1**2) 
                                    + rho_s**2 * self.K_E*self.K_F * (2*rho*self.coga2/den2 - (rho**2 * self.coga2 *(2*rho*(self.K_E-16*self.K_F*self.coga2) - 4*rho**3*self.coga2*(self.K_E-8*self.K_F*self.coga2)))/den2**2 ) 
                                    + 8*self.K_S*(np.arccos(1- rho**2 *2 *self.coga2)-np.pi + 2*self.gamma)  /np.sqrt(1-(1-rho**2 *2*self.coga2)**2)  *rho * self.coga2 
                                    )
    
    def W_rho_s(self, rho, rho_s, Gamma, theta_s):
        den2 = (1 - rho**2 * self.coga2 )* (8*self.K_F + rho**2 *(self.K_E - 8*self.K_F*self.coga2))
        return (rho**2 * self.coga2  /den2 * 2 * rho_s) * self.K_E * self.K_F

    def W_Gamma(self, rho, rho_s, Gamma, theta_s):
        return 0

    def W_theta_s(self, rho, rho_s, Gamma, theta_s):
        den1 = (rho**2 * self.coga2 * (self.K_E - 8*self.K_F*self.coga2) - self.K_E)
        return ((rho**2 * self.coga2 -1 ) /den1 * 2 * theta_s)  * self.K_E * self.K_F

    def W_rho_rho(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho: self.W_rho(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)

    def W_rho_rho_s(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho_s: self.W_rho(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho_s)


    def W_rho_theta_s(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, theta_s: self.W_rho(rho, rho_s, Gamma, theta_s), order=1)._x(0, theta_s)

    def W_rho_s_rho(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho: self.W_rho_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)

    def W_rho_s_rho_s(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho_s: self.W_rho_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho_s)

    def W_theta_s_rho(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho: self.W_theta_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, rho)

    def W_theta_s_theta_s(self, rho, rho_s, Gamma, theta_s):
        return Numerical_derivative(lambda t, rho: self.W_theta_s(rho, rho_s, Gamma, theta_s), order=1)._x(0, theta_s)
