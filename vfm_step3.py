"""
Version: final.3 (with shear + mix virtual fields, normalised)
Date: 2025-12-06
Author: mou guangjin + ChatGPT

Step 3 : Build interior VFM equations (x, y, shear, mix virtual fields),
         with per-equation normalisation of the virtual strain field.
"""

import numpy as np
import os

# ------------------- User settings -------------------
OUTPUT_DIR = "outputs"
NPZ_FILE   = os.path.join(OUTPUT_DIR, "coarse_strain_Q4_last10.npz")
THICKNESS  = 5.0
TRIM_LAYERS = 3

# ========== Virtual field switches ==========
# Control which virtual fields to include in the system
USE_VF_X     = True    # x-direction virtual field
USE_VF_Y     = True    # y-direction virtual field
USE_VF_SHEAR = False   # shear virtual field
USE_VF_MIX   = False   # mixed virtual field

OUT_NPZ_ALL = os.path.join(OUTPUT_DIR, "interior_system_all.npz")
OUT_CSV_ALL = os.path.join(OUTPUT_DIR, "interior_system_all.csv")

# Individual output files for each virtual field
OUT_NPZ_X     = os.path.join(OUTPUT_DIR, "interior_system_x.npz")
OUT_CSV_X     = os.path.join(OUTPUT_DIR, "interior_system_x.csv")
OUT_NPZ_Y     = os.path.join(OUTPUT_DIR, "interior_system_y.npz")
OUT_CSV_Y     = os.path.join(OUTPUT_DIR, "interior_system_y.csv")
OUT_NPZ_SHEAR = os.path.join(OUTPUT_DIR, "interior_system_shear.npz")
OUT_CSV_SHEAR = os.path.join(OUTPUT_DIR, "interior_system_shear.csv")
OUT_NPZ_MIX   = os.path.join(OUTPUT_DIR, "interior_system_mix.npz")
OUT_CSV_MIX   = os.path.join(OUTPUT_DIR, "interior_system_mix.csv")

# ========== load coarse strain data ==========
data = np.load(NPZ_FILE)

eps_xx = data["eps_xx"]
eps_yy = data["eps_yy"]
eps_xy = data["eps_xy"]

NX = int(data["NX"])
NY = int(data["NY"])
dx = float(data["dx_coarse"])
dy = float(data["dy_coarse"])
t  = THICKNESS
n_frames = int(data["n_frames"])

print(f"[INFO] Loaded coarse strain: n_frames={n_frames}, NX={NX}, NY={NY}")

# ========== Check virtual field settings ==========
vf_switches = {
    "x": USE_VF_X,
    "y": USE_VF_Y,
    "shear": USE_VF_SHEAR,
    "mix": USE_VF_MIX
}
active_vfs = [name for name, enabled in vf_switches.items() if enabled]
print(f"[INFO] Active virtual fields: {active_vfs}")
if not active_vfs:
    raise ValueError("At least one virtual field must be enabled!")

# ========== Q4 derivatives ==========
g = 1.0 / np.sqrt(3.0)
gauss_pts = [(-g,-g), (g,-g), (g,g), (-g,g)]

def q4_dN_dxi_eta(xi, eta):
    dN1_dxi = -0.25*(1-eta);  dN2_dxi =  0.25*(1-eta)
    dN3_dxi =  0.25*(1+eta);  dN4_dxi = -0.25*(1+eta)

    dN1_deta = -0.25*(1-xi);  dN2_deta = -0.25*(1+xi)
    dN3_deta =  0.25*(1+xi);  dN4_deta =  0.25*(1-xi)

    return np.array([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi]), \
           np.array([dN1_deta, dN2_deta, dN3_deta, dN4_deta])

invJ_x = 2.0 / dx
invJ_y = 2.0 / dy
V_cell = dx * dy * t

# ========== interior region ==========
i_min_calc = TRIM_LAYERS
i_max_calc = NX - 1 - TRIM_LAYERS
i_min = min(i_min_calc, i_max_calc)
i_max = max(i_min_calc, i_max_calc)

j_min_calc = TRIM_LAYERS
j_max_calc = NY - 1 - TRIM_LAYERS
j_min = min(j_min_calc, j_max_calc)
j_max = max(j_min_calc, j_max_calc)

print(f"[INFO] Interior cells: i in [{i_min}, {i_max}], j in [{j_min}, {j_max}]")
print(f"[INFO] Note: 3×3 patch for boundary interior cells will include cells in TRIM region")
print(f"[INFO]   Example: ic={i_min} patch includes i={i_min-1} (in TRIM), ic={i_max} patch includes i={i_max+1} (in TRIM)")

# ========== Virtual fields ==========
VIRTUAL_FIELDS_ALL = ["x", "y", "shear", "mix"]
VIRTUAL_FIELDS = [vf for vf in VIRTUAL_FIELDS_ALL if vf_switches[vf]]

print(f"[INFO] Building equations with virtual fields: {VIRTUAL_FIELDS}")

NODE_SCALE = 1.0 / 4.0

rows = []
meta  = []   # [frame, j_center, i_center, dir_idx_in_VIRTUAL_FIELDS]

# ========== Assemble equations (with normalisation) ==========
for k in range(n_frames):
    print(f"[INFO] Frame {k}: assembling...")

    for jc in range(j_min, j_max + 1):
        for ic in range(i_min, i_max + 1):

            center_nodes = {
                (ic,   jc),
                (ic+1, jc),
                (ic+1, jc+1),
                (ic,   jc+1)
            }

            for dir_idx, direction in enumerate(VIRTUAL_FIELDS):

                coeff_C11 = 0.0
                coeff_C22 = 0.0
                coeff_C66 = 0.0
                coeff_C12 = 0.0

                # 用于计算虚拟应变能范数的累积量
                acc_eps_xx_star_sq = 0.0
                acc_eps_yy_star_sq = 0.0
                acc_eps_xy_star_sq = 0.0

                # ---- 3×3 patch ----
                for j in range(jc - 1, jc + 2):
                    for i in range(ic - 1, ic + 2):

                        e1 = eps_xx[k, j, i]
                        e2 = eps_yy[k, j, i]
                        e6 = eps_xy[k, j, i]   # tensor shear strain

                        node_coords = [
                            (i,   j),
                            (i+1, j),
                            (i+1, j+1),
                            (i,   j+1)
                        ]

                        # ===== virtual displacement at 4 nodes =====
                        if direction == "x":
                            Vx_nodes = np.array(
                                [NODE_SCALE if nc in center_nodes else 0.0
                                 for nc in node_coords]
                            )
                            Vy_nodes = np.zeros(4)

                        elif direction == "y":
                            Vx_nodes = np.zeros(4)
                            Vy_nodes = np.array(
                                [NODE_SCALE if nc in center_nodes else 0.0
                                 for nc in node_coords]
                            )

                        elif direction == "shear":
                            pattern = np.array([1, -1, 1, -1], dtype=float)
                            Vx_nodes = NODE_SCALE * np.array([
                                pattern[idx] if nc in center_nodes else 0.0
                                for idx, nc in enumerate(node_coords)
                            ])
                            Vy_nodes = NODE_SCALE * np.array([
                                -pattern[idx] if nc in center_nodes else 0.0
                                for idx, nc in enumerate(node_coords)
                            ])

                        elif direction == "mix":
                            Vx_nodes = NODE_SCALE * np.array([1, 1, -1, -1]) * \
                                       np.array([1 if nc in center_nodes else 0.0
                                                 for nc in node_coords])
                            Vy_nodes = NODE_SCALE * np.array([1, -1, -1, 1]) * \
                                       np.array([1 if nc in center_nodes else 0.0
                                                 for nc in node_coords])

                        # ===== compute virtual strain (Gauss-averaged) =====
                        eps_xx_star = 0.0
                        eps_yy_star = 0.0
                        eps_xy_star = 0.0

                        for (xi, eta) in gauss_pts:
                            dN_dxi, dN_deta = q4_dN_dxi_eta(xi, eta)
                            dN_dx = dN_dxi * invJ_x
                            dN_dy = dN_deta * invJ_y

                            dvx_dx = np.dot(dN_dx, Vx_nodes)
                            dvx_dy = np.dot(dN_dy, Vx_nodes)
                            dvy_dx = np.dot(dN_dx, Vy_nodes)
                            dvy_dy = np.dot(dN_dy, Vy_nodes)

                            eps_xx_star += dvx_dx
                            eps_yy_star += dvy_dy
                            eps_xy_star += 0.5 * (dvx_dy + dvy_dx)

                        eps_xx_star /= 4.0
                        eps_yy_star /= 4.0
                        eps_xy_star /= 4.0

                        # ===== accumulate virtual strain "energy" =====
                        # 这里用简单的 L2 范数：eps_xx^2 + eps_yy^2 + eps_xy^2
                        acc_eps_xx_star_sq += (eps_xx_star ** 2) * V_cell
                        acc_eps_yy_star_sq += (eps_yy_star ** 2) * V_cell
                        acc_eps_xy_star_sq += (eps_xy_star ** 2) * V_cell

                        # ===== contribute to A (before normalisation) =====
                        coeff_C11 += V_cell * (e1 * eps_xx_star)
                        coeff_C22 += V_cell * (e2 * eps_yy_star)
                        # Voigt shear: sigma_xy = 2*C66*eps_xy → internal work = 2*eps_xy*eps_xy_star*C66
                        coeff_C66 += V_cell * (4.0 * e6 * eps_xy_star)
                        coeff_C12 += V_cell * (e2 * eps_xx_star + e1 * eps_yy_star)

                # ===== per-equation normalisation =====
                norm_eps_star = np.sqrt(
                    acc_eps_xx_star_sq + acc_eps_yy_star_sq + acc_eps_xy_star_sq
                )

                if norm_eps_star > 1e-14:
                    coeff_C11 /= norm_eps_star
                    coeff_C22 /= norm_eps_star
                    coeff_C66 /= norm_eps_star
                    coeff_C12 /= norm_eps_star
                else:
                    # 极端情况下虚拟场几乎为零，保留原系数（或可选择跳过）
                    pass

                rows.append([coeff_C11, coeff_C22, coeff_C66, coeff_C12])
                meta.append([k, jc, ic, dir_idx])

# ========== Save results ==========
A_int_all = np.array(rows)
b_int_all = np.zeros(A_int_all.shape[0])
meta      = np.array(meta, dtype=int)

print(f"[INFO] Interior system total equations (normalised) = {A_int_all.shape[0]}")
print(f"[INFO] Virtual fields used: {VIRTUAL_FIELDS}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.savez(
    OUT_NPZ_ALL,
    A=A_int_all,
    b=b_int_all,
    meta=meta,
    dx=dx,
    dy=dy,
    thickness=t,
    NX=NX,
    NY=NY,
    TRIM_LAYERS=TRIM_LAYERS,
    virtual_fields=np.array(VIRTUAL_FIELDS, dtype=object),
    USE_VF_X=USE_VF_X,
    USE_VF_Y=USE_VF_Y,
    USE_VF_SHEAR=USE_VF_SHEAR,
    USE_VF_MIX=USE_VF_MIX,
)

stack_all = np.column_stack([A_int_all, b_int_all])
np.savetxt(
    OUT_CSV_ALL,
    stack_all,
    delimiter=",",
    header="coeff_C11,coeff_C22,coeff_C66,coeff_C12,b(=0)",
    comments=""
)

# ========== Split and save individual virtual field results ==========
# meta[:, 3] contains dir_idx (0, 1, 2, 3 for x, y, shear, mix in VIRTUAL_FIELDS)

vf_output_config = {
    "x": (OUT_NPZ_X, OUT_CSV_X, USE_VF_X),
    "y": (OUT_NPZ_Y, OUT_CSV_Y, USE_VF_Y),
    "shear": (OUT_NPZ_SHEAR, OUT_CSV_SHEAR, USE_VF_SHEAR),
    "mix": (OUT_NPZ_MIX, OUT_CSV_MIX, USE_VF_MIX)
}

for vf_name, (npz_path, csv_path, is_enabled) in vf_output_config.items():
    if not is_enabled:
        continue  # Skip if this virtual field is not used
    
    # Find the index in VIRTUAL_FIELDS list
    if vf_name not in VIRTUAL_FIELDS:
        continue
    
    vf_idx_in_list = VIRTUAL_FIELDS.index(vf_name)
    
    # Extract equations corresponding to this virtual field
    mask = (meta[:, 3] == vf_idx_in_list)
    A_vf = A_int_all[mask, :]
    b_vf = b_int_all[mask]
    meta_vf = meta[mask, :]
    
    # Save NPZ
    np.savez(
        npz_path,
        A=A_vf,
        b=b_vf,
        meta=meta_vf,
        dx=dx,
        dy=dy,
        thickness=t,
        NX=NX,
        NY=NY,
        TRIM_LAYERS=TRIM_LAYERS,
        virtual_field=vf_name,
    )
    
    # Save CSV
    stack_vf = np.column_stack([A_vf, b_vf])
    np.savetxt(
        csv_path,
        stack_vf,
        delimiter=",",
        header="coeff_C11,coeff_C22,coeff_C66,coeff_C12,b(=0)",
        comments=""
    )
    
    print(f"[INFO] Saved {vf_name}-direction system: {A_vf.shape[0]} equations")
    print(f"       NPZ: {npz_path}")
    print(f"       CSV: {csv_path}")

print(f"\n[INFO] Saved NORMALISED interior system (ALL) to:")
print(f"       {OUT_NPZ_ALL}")
print(f"       {OUT_CSV_ALL}")
print(f"\n[INFO] Summary of virtual fields:")
print(f"       X-direction:  {'ON' if USE_VF_X else 'OFF'} ({np.sum(meta[:, 3] == VIRTUAL_FIELDS.index('x')) if 'x' in VIRTUAL_FIELDS else 0} equations)")
print(f"       Y-direction:  {'ON' if USE_VF_Y else 'OFF'} ({np.sum(meta[:, 3] == VIRTUAL_FIELDS.index('y')) if 'y' in VIRTUAL_FIELDS else 0} equations)")
print(f"       Shear:        {'ON' if USE_VF_SHEAR else 'OFF'} ({np.sum(meta[:, 3] == VIRTUAL_FIELDS.index('shear')) if 'shear' in VIRTUAL_FIELDS else 0} equations)")
print(f"       Mix:          {'ON' if USE_VF_MIX else 'OFF'} ({np.sum(meta[:, 3] == VIRTUAL_FIELDS.index('mix')) if 'mix' in VIRTUAL_FIELDS else 0} equations)")
print(f"       Total:        {A_int_all.shape[0]} equations")
print("[INFO] Done.")
