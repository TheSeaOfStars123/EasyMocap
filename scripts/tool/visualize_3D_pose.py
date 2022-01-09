'''
  @ Date: 2021/12/16 17:14
  @ Author: Zhao YaChen
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin


def renderBones():
   link = [[0,2],[0,3],[2,5],[3,6],[1,5],[1,6]
             ,[5,11],[11,15],[6,12],[12,16]
             ,[3,8],[8,14],[14,17], [2,7],[7,13],[13,18]
             ,[1,4],[4,9],[4,10]]
   for l in link:
       index1,index2 = l[0],l[1]
       ax.plot([xs[index1],xs[index2]], [ys[index1],ys[index2]], [zs[index1],zs[index2]], linewidth=1, label=r'$z=y=x$')

x_major_locator = MultipleLocator(0.1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(x_major_locator)
ax.zaxis.set_major_locator(x_major_locator)

# xs = [1.23449,1.26109,1.31214,1.12714,1.08315,1.35968,1.13569,1.33013,1.05112,1.21635,1.15264,1.45452,1.09412,1.35265, 1.0675,1.25544,1.01167, 0.951523,1.31394]
# ys = [ -0.565064,-0.550035,-0.495435,-0.612834,-0.324778,-0.460297,-0.615028,-0.514095,-0.602575,-0.410043,-0.453141,-0.450693,-0.746307,-0.543584,-0.664165,-0.328582, -0.53586,-0.614461,-0.420351]
# zs = [0.81514,1.35549, 0.810555, 0.824702,1.58366,1.33596,1.36724, 0.404219, 0.413127,1.56018,1.56505,1.07123,1.12532,-0.00409877,0.0153292,1.02795,1.03707,-0.0496544,-0.0499604]

xs = [0.000488868,0.0055941,-0.103808,0.10478,5.82369e-07,-0.157585,0.168773,-0.107429,0.124541,-0.0738106,0.0746384,-0.447876,0.439164,-0.0902739,0.0789466,-0.711255,0.710768,0.0858775,-0.0768233]
ys = [-0.445393,0.130922,-0.45544,-0.436001,0.289632,0.130455,0.131388,-0.851829,-0.858251,0.289369,0.295835,0.042134,0.0129227,-1.28674,-1.27954,0.0421627,0.0385358,-1.34135,-1.34326]
zs = [-0.00307689,-0.0251951,0.00318388,-0.00946845,0.116868,-0.0256311,-0.0247591,-0.0183414,-0.0174598,-0.0158212,-0.0154016,-0.0516878,-0.0637898,-0.058279,-0.0586063,-0.06351,-0.0644381,0.153596,0.151425]
renderBones()
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
plt.show()