import numpy as np
from enum import Enum

class BCType(Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    ROBIN = "Robin"

class Boundary_Conditions:
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
    


