import sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def parse_args():
    p = argparse.ArgumentParser(description="Run SKMD loop with configurable parameters.")
    p.add_argument("--fname", type=str,
                   help="Directory/prefix for plots and configs.")
    p.add_argument("--nsteps", type=int,
                   help="Number of steps used in simulation")
    p.add_argument("--initial_particle_positions", type=str,default=None,help="Path to initial particle positions. None for \{data_path\}/dump.lammpstrj")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder_path = args.fname
    data_path = f"{folder_path}/traj_data"

    nsteps = args.nsteps
    if args.initial_particle_positions is None:
        init_dump = read(f"{data_path}/dump.lammpstrj", index=":")
    else:
        init_dump = read(args.initial_particle_positions, index=":")
    nparticles = len(init_dump)
    print(nparticles, "particles")
    print("reading...")
    all_traj_full = [[] for i in range(nparticles)]
    pnum = 0
    last_points = []
    for st in tqdm(range(nsteps)):
        move_traj = read(f"{data_path}/{st}.lammpstrj", index=":")
        mace_desc_ndim = max(
            int(k[len("c_macedesc["):-1])
            for k in move_traj[0].arrays
            if k.startswith("c_macedesc[") and k.endswith("]")
        )
        for i in range(len(move_traj)):
            global_desc = []
            mask = (np.array(move_traj[i].get_chemical_symbols()) != "X")
            for dim in range(1, mace_desc_ndim + 1):
                global_desc.append(np.sum(move_traj[i].arrays[f"c_macedesc[{dim}]"][mask]))
                del move_traj[i].arrays[f"c_macedesc[{dim}]"]
            move_traj[i].info["mace_descriptor"] = global_desc
            move_traj[i].info["unbiased_energy"] = np.sum(move_traj[i].arrays["c_peat"])
            del move_traj[i].arrays["c_peat"]

        all_traj_full[pnum].extend(move_traj)
        last_points.append(move_traj[-1])
        pnum += 1
        if pnum == nparticles:
            pnum = 0

    print("writing...")
    for pnum in tqdm(range(nparticles)):
        write(f"{data_path}/ptraj_{pnum}.xyz", all_traj_full[pnum], format="extxyz")

    write(f"{folder_path}/final_points_processed.xyz", last_points, format="extxyz")