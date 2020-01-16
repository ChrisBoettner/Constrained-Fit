import numpy as np
from numpy import sin as s
from numpy import cos as c

# =============================================================================
# Data
# =============================================================================

def Enter_Data(Interpretation):
    if Interpretation == 1:
        # Interpretation 1: Lambda -> p pi-
        y         = np.array([1535,1.549,6.107,378,1.668,5.768,1.558,6.062])
        m         = np.array([938.272,139.570,1115.683])
        
        E         = np.array([np.sqrt(y[0]**2+m[0]**2),np.sqrt(y[3]**2+m[1]**2)
                    ])
        xi_inital = np.sqrt(E[0]**2+E[1]**2+2*E[0]*E[1]-m[2]**2)
        
  
        variances = np.hstack([np.diag([5184,3.6e-05,4.9e-05,324,0.0256,0.0196
                                        ]),np.zeros((6,2))])
        C_yy      = np.vstack([variances,[[0,0,0,0,0,0,5.29e-4,-1.3e-7],
                                          [0,0,0,0,0,0,-1.3e-7,2.11e-4]]])
    else:
        # Interpretation 2: K_S -> pi+ pi-
        y         = np.array([1479,1.552,6.111,378,1.668,5.768,1.558,6.062])
        m         = np.array([139.570,139.570,497.611])
        
        E         = np.array([np.sqrt(y[0]**2+m[0]**2),np.sqrt(y[3]**2+m[1]**2)
                    ])
        xi_inital = np.sqrt(E[0]**2+E[1]**2+2*E[0]*E[1]-m[2]**2)
  
        variances = np.hstack([np.diag([3600,3.6e-05,3.6e-05,324,0.0256,0.0196
                                        ]),np.zeros((6,2))])
        C_yy      = np.vstack([variances,[[0,0,0,0,0,0,5.29e-4,-1.3e-7],
                                          [0,0,0,0,0,0,-1.3e-7,2.11e-4]]])  
        
    Initial_Guess = Fitting(y,xi_inital,m,C_yy)
    return(Initial_Guess)

# =============================================================================
# Fitting Class
# =============================================================================

class Fitting():
    
    def __init__(self, y, xi_inital, m, C_yy):     
        self.m_d1 = m[0]
        self.m_d2 = m[1]
        self.m_V0 = m[2]        
        self.y = y
        self.C_yy = C_yy
        
        self.xi = xi_inital
        self.x = y
        self.lam = None
        self.xi_old = None
        
        self.L_old = None
        self.L = None
        
    def make_step(self):
        G_x     = self.__calc_Gx()
        G_xi    = self.__calc_Gxi()
        g       = self.__calc_g()
        r       = self.__calc_r(G_x,g)
        Sinv       = self.__calc_Sinv(G_x)
        
        self.__update_xi(G_xi,Sinv,r)
        self.__update_lambda(G_xi,Sinv,r)
        self.__update_x(G_x)
        
        self.L_old = self.L
        self.__calc_L(g)
        
    def calc_Var(self):
        G_x     = self.__calc_Gx()
        G_xi    = self.__calc_Gxi()
        Sinv       = self.__calc_Sinv(G_x)                

        A,B,U = calc_helper(G_x,G_xi,Sinv)
         
        C_xx_help = np.identity(8) - ((A - U* np.outer(B,B.T)) @ self.C_yy)
        C_xx  = self.C_yy @ C_xx_help
        C_xixi = U
        C_xxi  = - self.C_yy @ B * U              
        return(C_xx,C_xixi,C_xxi)
        
    def __calc_L(self,g):
        self.L = (self.y-self.x).T @ np.linalg.inv(self.C_yy) @ (self.y-self.x) + 2*self.lam.T@g
        return()
    
    def __update_xi(self,G_xi,Sinv,r):
        self.xi_old = self.xi
        self.xi = self.xi - 1/(G_xi.T @ Sinv @ G_xi) * G_xi.T @ Sinv @ r
        return()
        
    def __update_lambda(self,G_xi,Sinv,r):
        self.lam = Sinv @ (r + G_xi * (self.xi_old -self.xi))
        return()
    
    def __update_x(self,G_x):
        self.x = self.y - self.C_yy @ G_x.T @ self.lam
        return()
    
    def __calc_g(self):
        g_1 =  np.sqrt(self.x[0]**2+self.m_d1**2)+np.sqrt(self.x[3]**2+self.m_d2**2) -np.sqrt(self.xi**2+self.m_V0**2)
        g_2 = self.x[0]*s(self.x[1])*c(self.x[2])+self.x[3]*s(self.x[4])*c(self.x[5]) -self.xi*s(self.x[6])*c(self.x[7])
        g_3 = self.x[0]*s(self.x[1])*s(self.x[2])+self.x[3]*s(self.x[4])*s(self.x[5]) -self.xi*s(self.x[6])*s(self.x[7])
        g_4 = self.x[0]*c(self.x[1])+self.x[3]*c(self.x[4]) -self.xi*c(self.x[6])
        g   = np.array([g_1,g_2,g_3,g_4])
        return(g)
        
    def __calc_Gxi(self):
        G_xi_1 = self.xi/np.sqrt(self.xi**2+self.m_V0**2)
        G_xi_2 = s(self.x[6])*c(self.x[7])
        G_xi_3 = s(self.x[6])*s(self.x[7])
        G_xi_4 = c(self.x[6])
        G_xi = -np.array([G_xi_1,G_xi_2,G_xi_3,G_xi_4])
        return(G_xi)

    def __calc_Gx(self):
        G_x_1 = np.array([self.x[0]/np.sqrt(self.x[0]**2+self.m_d1**2),0,0,
                          self.x[3]/np.sqrt(self.x[3]**2+self.m_d2**2),0,0,
                          0,0])
        G_x_2 = np.array([s(self.x[1])*c(self.x[2]),self.x[0]*c(self.x[1])*c(self.x[2]),-self.x[0]*s(self.x[1])*s(self.x[2]),
                          s(self.x[4])*c(self.x[5]),self.x[3]*c(self.x[4])*c(self.x[5]),-self.x[3]*s(self.x[4])*s(self.x[5]),
                          -self.xi*c(self.x[6])*c(self.x[7]),self.xi*s(self.x[6])*s(self.x[7])])
        G_x_3 = np.array([s(self.x[1])*s(self.x[2]),self.x[0]*c(self.x[1])*s(self.x[2]),self.x[0]*s(self.x[1])*c(self.x[2]),
                          s(self.x[4])*s(self.x[5]),self.x[3]*c(self.x[4])*s(self.x[5]),self.x[3]*s(self.x[4])*c(self.x[5]),
                          -self.xi*c(self.x[6])*s(self.x[7]),-self.xi*s(self.x[6])*c(self.x[7])])
        G_x_4 = np.array([c(self.x[1]),-self.x[0]*s(self.x[1]),0,
                          c(self.x[4]),-self.x[3]*s(self.x[4]),0,
                          self.xi*s(self.x[6]),0])
        G_x = np.array([G_x_1,G_x_2,G_x_3,G_x_4])
        return(G_x)
    
    def __calc_Sinv(self, G_x):
        return(np.linalg.inv(G_x @ self.C_yy @ G_x.T))
        
    def __calc_r(self, G_x, g):
        return(g + G_x @ (self.y-self.x))
        
# =============================================================================
# External Help functions    
# =============================================================================
    
def calc_helper(G_x, G_xi, Sinv):
    A = G_x.T @ Sinv @ G_x
    B = G_x.T @ Sinv @ G_xi
    U = 1/(G_xi.T @ Sinv @ G_xi)
    return(A,B,U)       

    
# =============================================================================
# Calculation on Data
# =============================================================================
    
def Do_Fit(Initial_Guess, precision):
    Fit = Initial_Guess
    
    Fit.make_step()
    Fit.make_step()
    
    n = 0
    while(np.abs(Fit.L_old-Fit.L)>=precision):
        Fit.make_step()
        n=n+1
        if n>1000:
            print("n>1000 !")
            return()
    return(Fit, n)
    
    
if __name__ == "__main__":
    
    Interpretation = 1
    epsilon = 1e-8   
    Fit, n = Do_Fit(Enter_Data(Interpretation),epsilon)
    
    x = Fit.x
    xi = Fit.xi
    C_xx, C_xixi, C_xxi = Fit.calc_Var()
    x_err = np.sqrt(np.diag(C_xx))
    xi_err = np.sqrt(C_xixi)

    print(x)
    print(x_err)
    print(xi)
    print(xi_err)



    