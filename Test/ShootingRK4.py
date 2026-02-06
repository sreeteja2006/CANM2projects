import numpy as np
from typing import Callable

class ShootingRK4:

    def __init__(self,order : int):
        self.order = order
        

    def generateFunctionvector(self,f : callable) -> callable:
        def FunctionVector(x,Y):
            vec = np.zeros(self.order)
            for i in range(self.order - 1):
                vec[i] = Y[i+1]

            vec[-1] = f(x,Y)        
        return FunctionVector
        
    
    def rk_4(self):
        pass