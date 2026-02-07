import numpy as np
import tqdm

def RK4(self, u0):
    nx = int((self.xend - self.xstart) // self.h)
    U = np.zeros(shape=(nx+1,self.order),dtype= float)
    x=  np.zeros(nx+1)
    U[0] = u0
    x[0] = self.xstart
    def functionvector(x,U,f):
        return f(U,x)

    for i in tqdm.tqdm(range(1,nx+1)):
        k1 = self.h*functionvector(x[i-1],U[i-1],self.func)
        k2 = self.h*functionvector(x[i-1] + (self.h/2.0),U[i-1] + (k1/2.0),self.func)
        k3 = self.h*functionvector(x[i-1] + (self.h/2.0),U[i-1] + (k2/2.0),self.func)
        k4 = self.h*functionvector(x[i-1] + self.h,U[i-1] + k3,self.func)
        U[i] = U[i-1] + (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x[i] = x[i-1] + self.h
        if np.any(np.isnan(U[i])) or np.any(np.isinf(U[i])) or np.any(np.abs(U[i]) > 1e10):
            return U[:i+1], x[:i+1], True
    return U, x,False