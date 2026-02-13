import numpy as np

class Function_Generator:
    def __init__(self, F, Fy, Fyp):
        self.F = F
        self.Fy = Fy
        self.Fyp = Fyp

    def build_tridiagonal_terms(self):

        def fu(x, w, h, i):
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            return -1 + (h/2) * self.Fyp(xi, yi, ypi)

        def fl(x, w, h, i):
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            return -1 - (h/2) * self.Fyp(xi, yi, ypi)

        def fd(x, w, h, i):
            xi = x[i]
            yi = w[i]
            ypi = (w[i+1] - w[i-1]) / (2*h)

            return 2 + h**2 * self.Fy(xi, yi, ypi)

        return fu, fl, fd

    def build_residual_function(self):

        def residual(i, x, w, h):
            xi = x[i]
            yi = w[i]
            sip = (w[i+1] - w[i-1]) / (2*h)

            return (
                -w[i-1]
                + 2*w[i]
                - w[i+1]
                + h**2 * self.F(xi, yi, sip)
            )

        return residual