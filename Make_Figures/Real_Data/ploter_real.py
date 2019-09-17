## -*- coding: utf-8 -*-

from plotting import newfig, savefig
import matplotlib.pyplot as plt
import numpy as np

#aorta1_U = np.load("Aorta1_U.npy").item()
aorta1_A = np.load("Aorta1_A.npy").item()

aorta1_U = np.load("Aorta1_U.npy").item()
aorta2_U = np.load("Aorta2_U.npy").item()
aorta3_U = np.load("Aorta3_U.npy").item()
aorta4_U = np.load("Aorta4_U.npy").item()
carotid = np.load("LCommonCarotid_U.npy").item()
velo1_DG = np.load("velo_aorta1_ref.npy").item()
velo2_DG = np.load("velo_aorta2_ref.npy").item()
velo3_DG = np.load("velo_aorta3_ref.npy").item()
velo4_DG = np.load("velo_aorta4_ref.npy").item()

pinns_results = np.load("results_real_PINNs.npy").item()

t_p = pinns_results["Time"]
Velocity_aorta1   = pinns_results["Velocity_aorta1"]
Area_aorta1       = pinns_results["Area_aorta1"]
Pressure_aorta1   = pinns_results["Pressure_aorta1"]
Velocity_aorta2   = pinns_results["Velocity_aorta2"]
Area_aorta2       = pinns_results["Area_aorta2"]
Pressure_aorta2   = pinns_results["Pressure_aorta2"]
Velocity_aorta3   = pinns_results["Velocity_aorta3"]
Area_aorta3       = pinns_results["Area_aorta3"]
Pressure_aorta3   = pinns_results["Pressure_aorta3"]
Velocity_aorta4   = pinns_results["Velocity_aorta4"]
Area_aorta4       = pinns_results["Area_aorta4"]
Pressure_aorta4   = pinns_results["Pressure_aorta4"]

t_c = aorta1_U["t"]*1e-3

Tc = t_c.max()
t_r = velo1_DG["Time"]
aorta1U = aorta1_U["U"]*1e-2

aorta2U = aorta2_U["U"]*1e-2
aorta3U = aorta3_U["U"]*1e-2
aorta4U = aorta4_U["U"]*1e-2
carotidU = carotid["U"]*1e-2

u1_DG = velo1_DG["Velocity"]
u2_DG = velo2_DG["Velocity"]
u3_DG = velo3_DG["Velocity"]
u4_DG = velo4_DG["Velocity"]

fig1 = plt.figure(1,figsize=(10, 7), dpi=300, facecolor='w', frameon = False)
fig2 = plt.figure(2,figsize=(10, 7), dpi=300, facecolor='w', frameon = False)
fig3 = plt.figure(3,figsize=(10, 7), dpi=300, facecolor='w', frameon = False)
fig4 = plt.figure(4,figsize=(10, 7), dpi=300, facecolor='w', frameon = False)
fig5 = plt.figure(5,figsize=(10, 7), dpi=300, facecolor='w', frameon = False)
fig6 = plt.figure(6,figsize=(10, 7), dpi=300, facecolor='w', frameon = False)

fig1.clf()
fig2.clf()
fig3.clf()
fig4.clf()
fig5.clf()
fig6.clf()

ax1 = fig1.add_subplot(111)  
ax2 = fig2.add_subplot(111)  
ax3 = fig3.add_subplot(111)  
ax4 = fig4.add_subplot(111)   
ax5 = fig5.add_subplot(111)   
ax6 = fig6.add_subplot(111)   

ax1.plot(t_c, aorta1U,'b-',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 1')
ax1.plot(t_r, u1_DG,'r--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 1')
ax1.plot(t_p, Velocity_aorta1,'k--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 1')

ax2.plot(t_r, u1_DG,'r--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 1')
ax2.plot(t_r, u2_DG,'b--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 2')
ax2.plot(t_r, u3_DG,'y--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 3')
ax2.plot(t_r, u4_DG,'g--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 4')

ax3.plot(t_c, aorta3U,'b-',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 3')
ax3.plot(t_r, u3_DG,'r--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 3')
ax3.plot(t_p, Velocity_aorta3,'k--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 3')

ax4.plot(t_c, aorta4U,'b-',linewidth=2, markersize=0.5, label='Reference Pressure Vessel 4')
ax4.plot(t_r, u4_DG,'r--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 4')
ax4.plot(t_p, Velocity_aorta4,'k--',linewidth=2, markersize=0.5, label='Reference Velocity Vessel 4')

ax5.plot(t_c, carotidU,'b-',linewidth=2, markersize=0.5, label='Reference Pressure Carotid')
ax5.plot(t_r, u2_DG,'r--',linewidth=2, markersize=0.5, label='Reference Velocity Carotid')
ax5.plot(t_p, Velocity_aorta2,'k--',linewidth=2, markersize=0.5, label='Reference Velocity Carotid')

ax6.plot(t_p[1:,0], Pressure_aorta3[1:,0]/133.33,'r--',linewidth=2, markersize=0.5, label='Predicted Pressure Aorta 3')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()

filename1 = "aorta1_velocities.png"
filename2 = "comparative_velocities_DG.png"
filename3 = "aorta3_velocities.png"
filename4 = "aorta4_velocities.png"
filename5 = "carotid_velocities.png"
filename6 = "pressure_real.png"


fig1.savefig(filename1, format = 'png')
fig2.savefig(filename2, format = 'png')    
fig3.savefig(filename3, format = 'png')
fig4.savefig(filename4, format = 'png')    
fig5.savefig(filename5, format = 'png')
fig6.savefig(filename6, format = 'png')

plt.show()
