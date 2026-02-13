from Core.fdm_solver import FDM_Solver
from Core.Boundary_Conditions import *

b = np.array([[1, 1, -100],
              [1, 1, 0]])

bc_types = get_bc_type(b)
print(bc_types)


