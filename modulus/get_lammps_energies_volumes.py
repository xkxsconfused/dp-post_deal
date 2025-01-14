## 用于lammps计算体模量批量计算体系能量和体积

import os
import pandas as pd

    
data_e = {"" : []}
data_v = {"" : []}
list_structure = [""]
dir = ""
for j in list_structure:
    # os.chdir(f"{dir}/{j}")
    os.system(f"grep @@ {dir}/{j}/log.lammps > e-v.tmp")
    with open("e-v.tmp", 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 1:  # 偶数行（从0开始计数）
                # 提取Cu v后面的数字
                v_index = line.find('Volume ') + len('Volume ')
                v_value = float(line[v_index:].split(' ')[0])
                
                # 提取E--后面的数字
                e_index = line.find('E ') + len('E ')
                e_value = float(line[e_index:].strip())  # 使用strip()移除行尾的换行符
                
                # 将数据添加到字典中
                data_e[j].append(e_value)
                data_v[j].append(v_value)

# 创建DataFrame
df_e = pd.DataFrame(data_e)
df_v = pd.DataFrame(data_v)

# 将DataFrame保存为CSV文件
dir_csv_e = "energy-dp.csv"
dir_csv_v = "volume.csv"
df_e.to_csv(dir_csv_e, index=False)
df_v.to_csv(dir_csv_v, index=False)
