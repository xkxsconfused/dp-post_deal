import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.axislines import AxesZero
from matplotlib.pyplot import MultipleLocator
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

def major_formatter(x, pos):
    return f'{x:.0f}'

def set_plotparam(plot_params):
    fig, ax = plt.subplots(dpi=300)               # DPI设置
    # 导入自定义字体
    font_path = "/home/changsk/apps/fonts"
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    for file in font_files:
        font_manager.fontManager.addfont(file)
    # 全局设置字体及大小，设置公式字体即可，若要修改刻度字体，可在此修改全局字体
    config = {
        "mathtext.fontset":'stix',
        "font.family" : "Arial",
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
    
    # 创建x轴定位器，间隔2
    x_major_locator = MultipleLocator(plot_params['xmajor_locator'])
    # 创建y轴定位器，间隔5
    y_major_locator = MultipleLocator(plot_params['ymajor_locator'])
    # 设置x轴的间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置y轴的间隔
    ax.yaxis.set_major_locator(y_major_locator)
    
    # 修改有效小数位
    ax.xaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_major_formatter(major_formatter)
    
    ax.tick_params(labelsize=11)  #刻度字体大小
    # 设置刻度线向外还是向内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    return fig, ax

def ave(data):
    num = 10
    new_length = int(data.shape[0]/num)
    new_data = np.zeros((new_length,6))
    for i in range(new_length):
        for j in range(num):
            new_data[i] = new_data[i] + data[i*num+j]
        new_data[i] = new_data[i]/num
    return new_data
    
    
def rdf_file(filename,index):
    #c_myRDF[1]为group到原点的距离，即x轴，c_myRDF[2]为g(r)，c_myRDF[3]为coor( r )
    rdf_data = []
    head_row = 3 #最开始带#的行数
    n_row = 200 #每个step的行数
    start_row = head_row+index*(n_row+1)+1
    rdf_data = np.loadtxt(filename, skiprows=start_row, max_rows=n_row)
    return  rdf_data[:,[1,2]]

# "${N} ${T} ${V} ${pote} ${Etotal} ${Press}"
name = ""
dir_data = ""

data = np.loadtxt(dir_data+'data', delimiter=" ", dtype=float,skiprows=2)
data = ave(data)
index = np.arange(1,1000,2)
temperature = data[index,1]
V = data[index,2]
msd = np.loadtxt(dir_data+'msd.data',delimiter=" ", dtype=float,skiprows=2)
msd = msd[index,4]

rdfdata = []
temp_index_dict = {300:10,650:500,1000:1000}
for i,T in enumerate([300,650,1000]):
    rdfdata.append(rdf_file(dir_data+'tmp.rdf',temp_index_dict[T]))



plot_params = {
    "xs": 250,       # x轴起始坐标
    "xe": 1050,      # x轴终点坐标
    "ys": -100,      # y轴起始坐标
    "ye": 1100,      # y轴终点坐标
    "xlabel": r"Tempreture (K)",    # x轴标题
    "ylabel": r"MSD",               # y轴标题
    "xmajor_locator": 100,                # 设置x轴的间隔
    "ymajor_locator": 200,                # 设置y轴的间隔
}

# MSD-V-temperature关系
fig, ax = set_plotparam(plot_params)
# fig, ax= plt.subplots(dpi=300)

mp=590
ax.scatter(x=temperature, y=msd, color='r',s=10,alpha=0.7,edgecolors='k', label='MSD') # 标注点
ax.axvline(x=mp,color="red",linestyle='--')
ax.text(mp+20,0,f"Melting Point ≈ {mp} K",color='red')
ax2 = ax.twinx()
ax2.scatter(x=temperature, y=V, color='b', s=10, alpha=0.7, edgecolors='k', label='V')
ax2.set_ylabel(r'V(nm$^3$)',fontsize=12)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop={'size':15}, frameon=False)
plt.savefig('', bbox_inches='tight',dpi=300, transparent=True)

