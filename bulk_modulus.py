import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator
from matplotlib import font_manager 

def Murnaghan(params,vol):
    E0 = params[0]
    B0 = params[1]
    BP = params[2]
    V0 = params[3]
    E = E0 + (B0*vol/BP)*(((V0/vol)**BP)/(BP-1)+1) - V0*B0/(BP-1)
    return E

def residual(pars,y,x):
    err = y-Murnaghan(pars,x)
    return err 

def get_atom_num(dir, format):
    atoms=read(dir, format=format)
    atom_num = atoms.get_number_of_atoms()
    return atom_num
	
def nihebm(list1, list2):
    # list1为体积，list2为能量
    #多项式拟合
    a, b, c = np.polyfit(list1,list2,2)
    v0 = -b/(2*a)
    e0 = a*v0**2 + b*v0 + c
    b0 = 2*a*v0
    bp = 3.5
    x0 = [e0, b0, bp, v0]
    
    return x0
	
# 图形设置

def major_formatter_x(x, pos):
    return f'{x:.0f}'

def major_formatter_y(y, pos):
    return f'{y:.1f}'

def set_plotparam(plot_params):
    fig, ax = plt.subplots(figsize=(4.1,4),dpi=300)               # DPI设置
    # 全局设置字体及大小，设置公式字体即可，若要修改刻度字体，可在此修改全局字体
    # 设置字体路径
    font_path = ""
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    for file in font_files:
        font_manager.fontManager.addfont(file)
    config = {
        "mathtext.fontset":'stix',
        "font.family":"Arial",
        "font.serif": ['serif'],
        "font.size": 10,            # 字号
        'axes.unicode_minus': False # 处理负号，即-号
    }
    rcParams.update(config)
    ax.spines['right'].set_visible(True)         # 坐标轴可见性 
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)         # 坐标轴线框 
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_position(('data',plot_params['ys']))  # 移动x轴位置
    # plt.style.use
    # 设置坐标轴箭头
    #ax.plot(1, plot_params['ys'], ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot(plot_params['xs'], 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    plt.xlim(plot_params['xs'], plot_params['xe']+plot_params['xmajor_locator']*0.2)                               # x坐标轴刻度值范围
    plt.ylim(plot_params['ys'], plot_params['ye']+plot_params['ymajor_locator']*0.2)                               # y坐标轴刻度值范围
    plt.xlabel(plot_params['xlabel'],fontsize=12)                 # x坐标轴标题
    plt.ylabel(plot_params['ylabel'],fontsize=12)                 # y坐标轴标题
    
    # 创建x轴定位器，间隔
    x_major_locator = MultipleLocator(plot_params['xmajor_locator'])
    x_minor_locator = MultipleLocator(plot_params['xminor_locator'])
    # 创建y轴定位器，间隔
    y_major_locator = MultipleLocator(plot_params['ymajor_locator'])
    y_minor_locator = MultipleLocator(plot_params['yminor_locator'])
    # 设置x轴的间隔
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)
    # 设置y轴的间隔
    ax.yaxis.set_major_locator(y_major_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    # 修改有效小数位
    ax.xaxis.set_major_formatter(major_formatter_x)
    ax.yaxis.set_major_formatter(major_formatter_y)
    
    ax.tick_params(labelsize=11)  #刻度字体大小
    # 设置刻度线向外还是向内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    return fig, ax

def get_extremum(list1, list2):
    mx = np.max(np.concatenate((list1, list2)))
    mn = np.min(np.concatenate((list1, list2)))
    return mx, mn

# 结构列表
list_structure = []

# DP势函数拟合BM曲线
# 读取数据
# 提前用其他脚本整理的csv文件
df_e_dp = pd.read_csv("./energy_dp.csv")
df_e_dft = pd.read_csv("./energy_dft.csv")
df_v = pd.read_csv("./volume.csv")
# 读取data文件中的原子数目
j = list_structure[0]

# 对应结构的lammps-data文件
atom_num = get_atom_num("0.data", "lammps-data")

# 提取选定结构的能量和体积数据
list_energy_dp = df_e_dp[j]
list_energy_dft = df_e_dft[j]
list_volume = df_v[j]

#进行拟合
x0_dp = nihebm(list_volume, list_energy_dp)
murnpars_dp, ier_dp = leastsq(residual, x0_dp, args=(list_energy_dp, list_volume))
# _, _, e_cell_eam, e_atom_eam, x0_eam = nihebm(df_eam['v'], df_eam['e'])
# murnpars_eam, ier_eam = leastsq(residual, x0_eam, args=(e_cell_eam, v_cell))

#输出结果
print('Bulk Modulus:' + str(murnpars_dp[1])+ "eV/A^(-3)")
print('lattice constant:', murnpars_dp[3],"A")
print('Bulk Modulus:' + str(murnpars_dp[1]*1000/6.2415)+ "GPa")

# 绘EOS图
e_atom_dft = list_energy_dft / atom_num
e_atom_dp = list_energy_dp / atom_num
v_atom = list_volume / atom_num
max_v, min_v = get_extremum(v_atom, v_atom)
max_e, min_e = get_extremum(e_atom_dp, e_atom_dft)
xs = 0.92 * min_v
xe = 1.03 * max_v
ys = 1.01 * min_e
ye = 0.99 * max_e
plot_params = {
    "xs": xs,       # x轴起始坐标
    "xe": xe,        # x轴终点坐标
    "ys": ys,       # y轴起始坐标
    "ye": ye,        # y轴终点坐标
    "xlabel": r'Volume (Å$^\mathrm{\mathsf{3}}$/atom)',                    # x轴标题
    "ylabel": r'Energy (eV/atom)',                    # y轴标题
    "xmajor_locator": 4,                # 设置x轴的间隔
    "xminor_locator": 2,
    "ymajor_locator": 0.2,                # 设置y轴的间隔
    "yminor_locator": 0.1
                }
fig, ax = set_plotparam(plot_params)
v_mesh = np.linspace(np.min(list_energy_dft),np.max(list_volume),1000)
ax.scatter(v_atom,e_atom_dft,10,c='red',label='DFT')
ax.plot(v_mesh/atom_num,Murnaghan(murnpars_dp,v_mesh)/atom_num,linewidth=1,alpha=1,label='DP')
ax.text(0.6*xs+0.4*xe, 0.9*ye+0.1*ys,f"{j}")
ax.legend(frameon=False, loc="upper right")
fig.show()

# fig.savefig(f"{dir_file}/pic/{j}-dp.png", bbox_inches='tight', transparent=True)
