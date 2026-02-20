import numpy as np
import matplotlib.pyplot as plt


class StabilityComparison:
    def __init__(self, solver):
        self.solver = solver

    def compare_fdm_vs_shooting(
        self,
        fdm_tol=1e-10,
        fdm_max_iter=50,
        shoot_y0_init=0.0,
        shoot_yp0_init=1.0,
        shoot_h=5e-4,
        shoot_eps=1e-3,
        shoot_tol=1e-10,
        shoot_max_iter=30,
        verbose=False,
    ):
        _, it_fdm, hist_fdm = self.solver.solve_fdm(
            tol=fdm_tol,
            max_iter=fdm_max_iter,
            verbose=verbose,
        )

        y0_star, yp0_star, it_sh, ok_sh, hist_sh = self.solver.solve_shooting_newton(
            y0_init=shoot_y0_init,
            yp0_init=shoot_yp0_init,
            h=shoot_h,
            eps=shoot_eps,
            tol=shoot_tol,
            max_iter=shoot_max_iter,
            verbose=verbose,
        )

        out = {
            "FDM_iters": int(it_fdm),
            "FDM_cond2_final": float(hist_fdm["cond2"][-1]) if hist_fdm["cond2"] else float("inf"),
            "FDM_sigma_min_final": float(hist_fdm["sigma_min"][-1]) if hist_fdm["sigma_min"] else float("inf"),
            "FDM_inv_amp_max_final": float(hist_fdm["inv_amp_max"][-1]) if hist_fdm["inv_amp_max"] else float("inf"),
            "SHOOT_ok": bool(ok_sh),
            "SHOOT_iters": int(it_sh),
            "SHOOT_y0": float(y0_star) if np.isfinite(y0_star) else np.nan,
            "SHOOT_yp0": float(yp0_star) if np.isfinite(yp0_star) else np.nan,
            "SHOOT_cond2_final": float(hist_sh["cond2"][-1]) if hist_sh["cond2"] else float("inf"),
            "SHOOT_sigma_min_final": float(hist_sh["sigma_min"][-1]) if hist_sh["sigma_min"] else float("inf"),
            "SHOOT_inv_amp_max_final": float(hist_sh["inv_amp_max"][-1]) if hist_sh["inv_amp_max"] else float("inf"),
        }

        return out, hist_fdm, hist_sh

def plot_both_cond2(hist_fdm, hist_sh, out_path="outputs/plots/cond2_FDM_vs_SHOOT.png"):
    kf = np.arange(len(hist_fdm["cond2"]), dtype=float)
    ks = np.arange(len(hist_sh["cond2"]), dtype=float)

    plt.figure(figsize=(10, 6))
    if len(kf):
        plt.semilogy(kf, hist_fdm["cond2"], marker="o", label="FDM cond2(J)")
    if len(ks):
        plt.semilogy(ks, hist_sh["cond2"], marker="o", label="Shooting cond2(J)")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\kappa_2(J)$")
    plt.title("Condition number: FDM vs Shooting")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_both_sigma_min(hist_fdm, hist_sh, out_path="outputs/plots/sigmaMin_FDM_vs_SHOOT.png"):
    kf = np.arange(len(hist_fdm["sigma_min"]), dtype=float)
    ks = np.arange(len(hist_sh["sigma_min"]), dtype=float)

    plt.figure(figsize=(10, 6))
    if len(kf):
        plt.semilogy(kf, hist_fdm["sigma_min"], marker="o", label="FDM sigma_min(J)")
    if len(ks):
        plt.semilogy(ks, hist_sh["sigma_min"], marker="o", label="Shooting sigma_min(J)")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\sigma_{\min}(J)$")
    plt.title("Smallest singular value: FDM vs Shooting")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
