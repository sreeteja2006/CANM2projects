import numpy as np
from enum import Enum

class BCType(Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    ROBIN = "Robin"

class Boundary_Conditions:
    def __init__(self, BC: np.ndarray):
        self.BC = BC
        self.bc_types = self.get_bc_type(BC)


    @staticmethod
    def get_bc_type(BC: np.ndarray) -> list[BCType]:
        """
        BC format:
        For each boundary:
            a*y + b*y' + c= 0
        BC is shape (2, 3):
            [[a0, b0, c0],
            [a1, b1, c1]]
        """

        bc_types = []

        for i in range(2):
            a, b, _ = BC[i]

            if b == 0:
                bc_types.append(BCType.DIRICHLET)
            elif a == 0:
                bc_types.append(BCType.NEUMANN)
            else:
                bc_types.append(BCType.ROBIN)

        return bc_types

    def build_left_bc_jac(self, Fy, Fyp, x0):

        a, b, c = self.BC[0]

        if self.bc_types[0] == BCType.DIRICHLET:
            return lambda w, h: (a, 0.0)

        else:

            def jac(w, h):

                y0 = w[0]
                Yp = (-c - a*y0) / b

                d0 = (
                    2*(1 - h*a/b)
                    + h**2 * Fy(x0, y0, Yp)
                    - h**2 * (a/b) * Fyp(x0, y0, Yp)
                )

                u0 = -2.0

                return d0, u0

            return jac


    def build_right_bc_jac(self, Fy, Fyp, xN):

        a, b, c = self.BC[1]

        if self.bc_types[1] == BCType.DIRICHLET:
            return lambda w, h: (0.0, a)
        else:

            def jac(w, h):

                yNp1 = w[-1]
                yp = (-c - a*yNp1) / b

                dNp1 = (
                    2*(1 + h*a/b)
                    + h**2 * Fy(xN, yNp1, yp)
                    - h**2 * (a/b) * Fyp(xN, yNp1, yp)
                )

                lNp1 = -2.0

                return lNp1, dNp1

            return jac

    def build_left_bc_res(self, F, x0):

        a, b, c = self.BC[0]

        if self.bc_types[0] == BCType.DIRICHLET:
            return lambda w, h: a*w[0] + c

        else:

            def res(w, h):

                y0 = w[0]
                y1 = w[1]

                yp = (-c - a*y0) / b

                return (
                    2*(1 - h*a/b)*y0
                    - 2*y1
                    + h**2 * F(x0, y0, yp)
                    - 2*h *c/b
                )

            return res


    def build_right_bc_res(self, F, xN):

    
        a, b, c = self.BC[1]

        if self.bc_types[1] == BCType.DIRICHLET:
            c = self.BC[1][2]
            return lambda w, h: a*w[-1] + c
        else:
            
            def res(w, h):

                yNp1 = w[-1]
                yN = w[-2]

                yp = (-c - a*yNp1) / b

                return (
                    2*(1 + h*a/b)*yNp1
                    - 2*yN
                    + h**2 * F(xN, yNp1, yp)
                    + 2*h *c/b
                )

            return res