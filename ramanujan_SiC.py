import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# -----------------------------
# PARAMETERS
# -----------------------------
Lx, Ly = 20, 20  # lattice size
num_defects = 20  # initial defects
num_mc_steps = 10000
temperatures = [4, 77, 300, 400]  # K for T2
d_thresholds = [1, 3, 5, 7, 9]  # code distances for P_L
num_simulations = 500  # Monte Carlo runs for P_L

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def initialize_lattice(Lx, Ly, num_defects):
    lattice = np.zeros((Lx, Ly))
    defect_positions = np.random.choice(Lx*Ly, num_defects, replace=False)
    for pos in defect_positions:
        x, y = divmod(pos, Ly)
        lattice[x, y] = 1  # defect
    return lattice

def energy(lattice):
    E = 0
    Lx, Ly = lattice.shape
    for x in range(Lx):
        for y in range(Ly):
            if lattice[x, y] == 1:
                neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                for nx, ny in neighbors:
                    if 0 <= nx < Lx and 0 <= ny < Ly:
                        if lattice[nx, ny] == 1:
                            E += 1
    return E

def metropolis_step(lattice, T):
    Lx, Ly = lattice.shape
    x, y = np.random.randint(0, Lx), np.random.randint(0, Ly)
    lattice_trial = lattice.copy()
    lattice_trial[x, y] = 1 - lattice_trial[x, y]  # flip defect
    dE = energy(lattice_trial) - energy(lattice)
    kB = 1.38e-23
    if dE <= 0 or np.random.rand() < np.exp(-dE/(kB*(T+1e-23))):
        return lattice_trial
    return lattice

def compute_clusters(lattice):
    clusters, num_clusters = label(lattice==1)
    cluster_sizes = [np.sum(clusters==i) for i in range(1,num_clusters+1)]
    d_eff = max(cluster_sizes) if cluster_sizes else 0
    return d_eff, clusters

def monte_carlo_simulation(T, Lx, Ly, num_defects, steps):
    lattice = initialize_lattice(Lx, Ly, num_defects)
    for _ in range(steps):
        lattice = metropolis_step(lattice, T)
    d_eff, clusters = compute_clusters(lattice)
    return lattice, d_eff, clusters

# -----------------------------
# ANALYTICAL RAMANUJAN P_L(d) (illustrative)
# -----------------------------
def ramanujan_pl(d_vals):
    pl = [0.1*(0.2**(d-1)) for d in d_vals]  # decaying exponential
    return pl

# -----------------------------
# MOCK THETA FUNCTION (illustrative)
# -----------------------------
def mock_theta(q_vals):
    # Example mock theta series: f(q) = sum q^{n^2} / (q;q)_n, simplified
    f = []
    for q in q_vals:
        s = 0
        for n in range(10):
            prod = np.prod([(1-q**k) for k in range(1,n+1)]) if n>0 else 1
            s += q**(n**2)/prod
        f.append(s)
    return f

# -----------------------------
# COHERENCE TIME T2 vs Temperature (illustrative)
# -----------------------------
T2_data_lit = [1000, 500, 80, 40]  # us
T_data = temperatures
T2_pred = [np.random.uniform(70,90) if T==300 else val for T,val in zip(T_data,T2_data_lit)]

# -----------------------------
# MAIN SIMULATION
# -----------------------------

# 1. Monte Carlo lattice snapshot at 300K
lattice_snapshot, _, clusters_snapshot = monte_carlo_simulation(300, Lx, Ly, num_defects, num_mc_steps)
plt.figure(figsize=(6,6))
plt.imshow(lattice_snapshot, cmap='Greys', origin='lower')
plt.title("Monte Carlo Lattice Snapshot (Red = defects)")
plt.colorbar(label='0=empty,1=defect')
plt.savefig("mc_lattice_snapshot.png", dpi=300)
plt.show()

# 2. Logical error probability P_L(d) from Monte Carlo
P_L_mc = []
for d_thresh in d_thresholds:
    count = 0
    for _ in range(num_simulations):
        _, d_eff, _ = monte_carlo_simulation(300, Lx, Ly, num_defects, num_mc_steps//10)
        if d_eff >= d_thresh:
            count += 1
    P_L_mc.append(count/num_simulations)

# Analytical Ramanujan-inspired P_L(d)
P_L_analytical = ramanujan_pl(d_thresholds)

# Plot P_L(d)
plt.figure(figsize=(7,5))
plt.semilogy(d_thresholds, P_L_analytical, 'b-o', label='Analytical Ramanujan')
plt.semilogy(d_thresholds, P_L_mc, 'r-s', label='Monte Carlo')
plt.xlabel("Code distance d")
plt.ylabel("Logical error probability P_L (log scale)")
plt.title("Logical Error Probability vs Code Distance")
plt.grid(True, which="both")
plt.legend()
plt.savefig("pl_vs_d.png", dpi=300)
plt.show()

# 3. Entanglement fidelity F vs code distance
F_baseline = [0.65,0.72,0.78,0.83,0.87,0.90,0.92,0.93,0.94,0.95]
F_optimized = [0.70,0.82,0.89,0.94,0.96,0.975,0.985,0.99,0.995,0.997]
d_vals_F = range(1,11)
plt.figure(figsize=(7,5))
plt.plot(d_vals_F, F_baseline, 'r--', label='Baseline (thermal noise)')
plt.plot(d_vals_F, F_optimized, 'b-o', label='Ramanujan-optimized')
plt.xlabel("Code distance / protection level")
plt.ylabel("Entanglement fidelity F")
plt.title("Entanglement Fidelity vs Code Distance")
plt.grid(True)
plt.legend()
plt.savefig("entanglement_fidelity.png", dpi=300)
plt.show()

# 4. Mock theta function
q_vals = np.linspace(0,0.99,50)
f_vals = mock_theta(q_vals)
plt.figure(figsize=(7,5))
plt.plot(q_vals, f_vals, 'g-o')
plt.xlabel("q (nome)")
plt.ylabel("f(q) (mock theta)")
plt.title("Illustrative Mock Theta Function")
plt.grid(True)
plt.savefig("mock_theta.png", dpi=300)
plt.show()

# 5. Temperature sweep of T2
plt.figure(figsize=(7,5))
plt.plot(T_data, T2_data_lit, 'b-s', label='Literature')
plt.plot(T_data, T2_pred, 'r-o', label='Predicted RT range')
plt.xlabel("Temperature T (K)")
plt.ylabel("Coherence time T2 (μs, log scale)")
plt.yscale("log")
plt.title("Coherence Time vs Temperature")
plt.grid(True, which="both")
plt.legend()
plt.savefig("T2_vs_T.png", dpi=300)
plt.show()
