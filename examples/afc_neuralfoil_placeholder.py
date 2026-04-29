import aerosandbox as asb
import aerosandbox.numpy as np


airfoil = asb.Airfoil("naca4412")
alpha = np.linspace(-4, 14, 10)

baseline = airfoil.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=1e6,
    mach=0.05,
)

afc = airfoil.get_aero_from_afc_neuralfoil(
    alpha=alpha,
    Re=1e6,
    mach=0.05,
    C_mu=0.02,
    x_jet=0.10,
    theta_jet=30,
)

print("alpha:", alpha)
print("Baseline CL:", baseline["CL"])
print("AFC CL:", afc["CL"])
print("AFC dCL:", afc["dCL_afc"])
print("AFC analysis confidence:", afc["analysis_confidence"])
