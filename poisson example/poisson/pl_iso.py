import scipy.io as sio
import sys
import numpy as np
import pylab as p
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display
import os  # <--- 必须导入 os 模块

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})

plt.interactive(True)
plt.close('all')

# --- 核心修复：获取当前脚本所在的文件夹路径 ---
if '__file__' in globals():
   script_dir = os.path.dirname(os.path.abspath(__file__))
else:
   script_dir = os.getcwd()

# 构建文件的绝对路径
file_x = os.path.join(script_dir, "x2d.dat")
file_y = os.path.join(script_dir, "y2d.dat")
file_u = os.path.join(script_dir, "u2d_saved.npy") # 这个文件也需要加路径！

print(f"正在读取文件: {file_x}")

viscos=3.57E-5

# 使用绝对路径加载数据
try:
    datax= np.loadtxt(file_x)
    x=datax[0:-1]
    ni=int(datax[-1])

    datay= np.loadtxt(file_y)
    y=datay[0:-1]
    nj=int(datay[-1])
    
    # 加载计算结果
    u2d=np.load(file_u)
    
except FileNotFoundError as e:
    print(f"\n错误: 找不到文件。请确认所有 .dat 和 .npy 文件都在文件夹:\n{script_dir}")
    print(f"详细错误: {e}")
    sys.exit(1)

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])


########################################## iso u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.pcolormesh(xp2d,yp2d,u2d, cmap=plt.get_cmap('hot'),shading='gouraud')

plt.text(0.8,2.2,'$U=1$')
plt.axis('equal')
#plt.colorbar()
plt.axis('off')
plt.box(on=None)

# 保存图片也建议加上路径，防止不知道存哪去了
save_path_iso = os.path.join(script_dir, 'u_iso-poisson.png')
plt.savefig(save_path_iso, bbox_inches='tight')
print(f"图片已保存: {save_path_iso}")


########################################## grid
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

#%%%%%%%%%%%%%%%%%%%%% grid
for i in range(0,ni+1):
   plt.plot(x2d[i,:],y2d[i,:],'k-')

for j in range(0,nj+1):
   plt.plot(x2d[:,j],y2d[:,j],'k-')

plt.axis('equal')
plt.axis('off')
plt.text(0.8,2.2,'$U=1$')
plt.box(on=None)

save_path_grid = os.path.join(script_dir, 'grid.png')
plt.savefig(save_path_grid, bbox_inches='tight')
print(f"图片已保存: {save_path_grid}")

plt.show() # 最后加上 show 以便在窗口查看