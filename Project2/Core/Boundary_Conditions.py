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

    def build_left_bc_jac(self):
        if self.bc_types[0] == BCType.DIRICHLET:
            return lambda w, h: (1.0, 0.0)

    def build_right_bc_jac(self):
        if self.bc_types[1] == BCType.DIRICHLET:
            return lambda w, h: (0.0, 1.0)

    def build_left_bc_res(self):
        if self.bc_types[0] == BCType.DIRICHLET:
            c = self.BC[0][2]
            return lambda w, h: w[0] + c

    def build_right_bc_res(self):
        if self.bc_types[1] == BCType.DIRICHLET:
            c = self.BC[1][2]
            return lambda w, h: w[-1] + c