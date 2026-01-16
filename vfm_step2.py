"""
Version: final.1
Date: 2025-11-19
Author: mou guangjin
Email: mouguangjin@ust.hk

Step 2: This script is used to build the half domain VFM equations.

Input file: coarse strain data (10*10 for each frame)+foece_displacement.csv
Output file: half domain VFM equations

To do: replace the input file "force_displacement.csv" 
"""

import numpy as np
import os

# ------------------- User settings -------------------
NPZ_FILE   = "outputs/coarse_strain_Q4_last10.npz"
FORCE_CSV  = "inputs/force_displacement.csv"   # force file
THICKNESS  = 5  # sample thickness (mm)
# Note: TRIM_LAYERS is not used for half-domain equations.
# Half-domain equations use ALL cells because Fy is global reaction force.
# Boundary exclusion should only be applied to interior equations.

OUT_NPZ    = "outputs/half_domain_multiline_system.npz"
OUT_CSV    = "outputs/half_domain_multiline_system.csv"

# ========== Step 1: load coarse strain data ==========
data = np.load(NPZ_FILE)

eps_xx = data["eps_xx"]   # (n_frames, NY, NX)
eps_yy = data["eps_yy"]
eps_xy = data["eps_xy"]

NX = int(data["NX"])
NY = int(data["NY"])

dx = float(data["dx_coarse"])
dy = float(data["dy_coarse"])
t  = THICKNESS

n_frames = int(data["n_frames"])
print(f"[INFO] Loaded coarse strain from {NPZ_FILE}")
print(f"[INFO] n_frames = {n_frames}, NX = {NX}, NY = {NY}")

# ========== load reaction force Fy ==========
force_data = np.loadtxt(FORCE_CSV, delimiter=",", skiprows=1)
Fy_all = force_data[:, 3]   # read force_y_N

n_force = len(Fy_all)
print(f"[INFO] Loaded {n_force} force entries from {FORCE_CSV}")

if n_force < n_frames:
    raise RuntimeError("force_displacement.csv 的行数少于 npz 中的帧数，无法对齐。")

# align the last n_frames
Fy_list = Fy_all[-n_frames:]
print(f"[INFO] Using last {n_frames} force entries to match coarse strain frames.")

# ========== choose cut lines (1/4, 1/2, 3/4) ==========
# define three cut lines
j_cuts = [
    NY//4 - 1,      # 1/4 line
    NY//2 - 1,      # 1/2 line (center)
    3*NY//4 - 1,    # 3/4 line
]
print(f"[INFO] Using {len(j_cuts)} cut lines: {j_cuts}")
print(f"[INFO]   j_cut[0] = {j_cuts[0]} (1/4 line)")
print(f"[INFO]   j_cut[1] = {j_cuts[1]} (1/2 line, center)")
print(f"[INFO]   j_cut[2] = {j_cuts[2]} (3/4 line)")

# ========== Note on boundary cells ==========
# IMPORTANT: For half-domain equations, we should use ALL cells, not exclude boundaries.
# This is because Fy is the global reaction force, and the VFM requires:
#   Internal work (over entire domain) = External work (Fy * virtual displacement)
# If we exclude boundary cells from integration but Fy is global, the equation becomes unbalanced.
# Therefore, we use all NX cells for half-domain equations.
#
# Boundary exclusion should only be applied to INTERIOR equations (where b=0).
#
print(f"[INFO] Using ALL {NX} cells for half-domain equations (boundary cells included)")
print(f"[INFO]   Reason: Fy is global reaction force, equation must balance over entire domain")

# ========== virtual strain (unit virtual displacement) ==========
eps22_star_unit = 1.0 / dy

# ========== print equations and build A, b ==========
print("\n================ HALF DOMAIN EQUATIONS (3 CUT LINES) ================\n")

# each frame has 3 equations (corresponding to 3 cut lines), total n_frames * 3 equations
n_equations = n_frames * len(j_cuts)
A = np.zeros((n_equations, 2))
b = np.zeros(n_equations)
meta = []  # record the corresponding [frame, j_cut] for each equation

eq_idx = 0

for k in range(n_frames):
    print(f"\n{'='*70}")
    print(f"--- Frame {k} ---")
    print(f"{'='*70}")

    Fy = Fy_list[k]

    for cut_idx, j_cut in enumerate(j_cuts):
        print(f"\n  Cut line {cut_idx+1}/3: j_cut = {j_cut} ({'1/4' if cut_idx==0 else '1/2' if cut_idx==1 else '3/4'} line)")

        exx_row = eps_xx[k, j_cut, :]
        eyy_row = eps_yy[k, j_cut, :]

        V_i = dx * dy * t

        # Use ALL cells (including boundaries) for half-domain equations
        # Fy is the global reaction force and must balance with internal work over entire domain

        # print the contribution of each cell (optional, if output too many can be commented out)
        if NX <= 10:  # only print when the cell number is not too many
            for i in range(NX):
                print(
                    f"    Cell {i}:  V_i*( "
                    f"C22*({eyy_row[i]:.6e})*({eps22_star_unit:.6e})"
                    f" + C12*({exx_row[i]:.6e})*({eps22_star_unit:.6e}) )"
                )

        # Sum over ALL cells (including boundaries)
        coeff_C22 = np.sum(V_i * eyy_row * eps22_star_unit)
        coeff_C12 = np.sum(V_i * exx_row * eps22_star_unit)

        print(f"\n  Full equation (symbolic):")
        print(f"    Σ_i V_i*( C22*eps22_bar[i]*eps22_star + C12*eps11_bar[i]*eps22_star ) = Fy")
        print(f"    (summed over ALL {NX} cells including boundaries)")
        print(f"  Numerical form:")
        print(f"    ({coeff_C22:.6e})*C22  +  ({coeff_C12:.6e})*C12  =  {Fy:.6e}")

        # fill in A, b
        A[eq_idx, 0] = coeff_C22
        A[eq_idx, 1] = coeff_C12
        b[eq_idx]    = Fy
        meta.append([k, j_cut, cut_idx])

        eq_idx += 1

print(f"\n{'='*70}")
print(f"[INFO] Total equations: {n_equations} ({n_frames} frames × {len(j_cuts)} cut lines)")
print(f"{'='*70}\n")

# ========== save A, b ==========
os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

meta_array = np.array(meta, dtype=int)  # [frame, j_cut, cut_idx]

np.savez(
    OUT_NPZ,
    A=A,
    b=b,
    meta=meta_array,
    j_cuts=np.array(j_cuts),
    dx=dx,
    dy=dy,
    thickness=t,
    n_frames=n_frames,
    n_cut_lines=len(j_cuts),
)

print(f"[INFO] Saved A,b system to {OUT_NPZ}")
print(f"[INFO]   A shape: {A.shape} ({n_frames} frames × {len(j_cuts)} cut lines = {n_equations} equations)")
print(f"[INFO]   b shape: {b.shape}")
print(f"[INFO]   Cut lines: {j_cuts}")
print(f"[INFO]   Using ALL {NX} cells per cut line (boundary cells included)")

stack = np.column_stack([A, b])
np.savetxt(
    OUT_CSV,
    stack,
    delimiter=",",
    header="coeff_C22,coeff_C12,Fy",
    comments="",
)

print(f"[INFO] Saved CSV to {OUT_CSV}")
print("[INFO] Done.")
