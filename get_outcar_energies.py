## 用于vasp计算体模量的批量提取OUTCAR中的能量

from ase.io import read
import pandas as pd

def get_outcar_energy(dir):
    atoms = read(f"{dir}")
    energy = atoms.get_potential_energy()
    return energy

# 能量列表
data = {"" : []}
# OUTCAR目录
dir = ""
list_structure = [""]
# 提取能量
for j in list_structure:
    for i in range(1,21):
        energy = get_outcar_energy(f"{dir}/{j}/{i}/OUTCAR")
        data[j].append(energy)
# 生成pd
df = pd.DataFrame(data)
csv_name = "energy-dft.csv"
df.to_csv(f"{dir}/pic/{csv_name}", index=False)
