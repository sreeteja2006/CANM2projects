import numpy as np
import matplotlib.pyplot as plt
from core.core import odesolver
from methods.RK_4 import RK4
from methods.RK_2 import RK2
from methods.ABM_2 import ABM2
from methods.ABM_4 import ABM4


print("Welcome to ODE Solver 67")
print("/n Fuck my life")
def getinput():
    order = int(input("\nEnter the order of the ODE (1 or 2): "))


    print("\n Enter the function in form of f(u,x) = [u[0],u[1],....,f(u,x)]\n")
    print("\n For example, for u'' + 2u' + 3u = 0, you can enter: [u[1], -2*u[1] - 3*u[0]]\n")

    funcstr = input("f(u,x) = [")
    funcstr.rstrip("]")
    func = eval("lambda u,x: np.array([" + funcstr + "])")


    xstart = float(input("Enter the start value of x: "))
    xend = float(input("Enter the end value of x: "))
    h = float(input("Enter the step size h: "))


    print("\n Enter boundary conditions in the form of ay+by'+c=0:")
    print("\n At xstart:")
    a0 = float(input("a: "))
    b0 = float(input("b: "))
    c0 = float(input("c: "))
    print("\n At xend:")
    a1 = float(input("a: "))
    b1 = float(input("b: "))
    c1 = float(input("c: "))

    print('\n give the two guesses for shooting method:')
    guess1 = float(input("guess 1: "))
    guess2 = float(input("guess 2: "))

    guess = [guess1, guess2]

    methods = {
        1: RK4,
        2: RK2,
        3: ABM2,
        4: ABM4
    }

    print("\n Choose the method to solve the ODE:")
    print("1. Runge-Kutta 4th order (RK4)")
    print("2. Runge-Kutta 2nd order (RK2)")
    print("3. Adams-Bashforth-Moulton 2nd order (ABM2)")
    print("4. Adams-Bashforth-Moulton 4th order (ABM4)")

    method_choice = int(input("Enter the number corresponding to the method: "))
    method = methods.get(method_choice, RK4)

    return order, func, xstart, xend, h, (a0, b0, c0), (a1, b1, c1), guess, method



        
        