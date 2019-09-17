# -*- coding: utf-8 -*-

from plotting import newfig, savefig
import matplotlib.pyplot as plt
import numpy as np

results_un = np.load("results_bif_unnorm.npy").item()
results_bif = np.load("results_bif.npy").item()
results_RCR2 = np.load("vessel_2_bif.npy").item()
results_RCR3 = np.load("vessel_3_bif.npy").item()

T_f = results_un["Time_f"]
t = results_un["Time_t"]
T_plot= t
p_pred_un = results_un["PredPress1"]
u_pred_un = results_un["PredVel1"]
p_pred = results_bif["PredPressl_2"]
u_pred = results_bif["PredVel_2"]
p_test = results_bif["TestPress_2"]
u_test = results_bif["TestVel_2"]
p_RCR_2 = results_RCR2["Pressure"].T
u_RCR_2 = results_RCR2["Velocity"].T
p_RCR_3 = results_RCR3["Pressure"].T
u_RCR_3 = results_RCR3["Velocity"].T

fig1 = plt.figure(1,figsize=(15, 10), dpi=300, facecolor='w', frameon = False)
fig2 = plt.figure(2,figsize=(15, 10), dpi=300, facecolor='w', frameon = False)
fig1.clf()
fig2.clf()

ax1 = fig1.add_subplot(111)  
ax2 = fig2.add_subplot(111)  

ax1.plot(t, u_test,'r-',linewidth=2, markersize=0.5, label = 'Discontinuous Galerkin')
ax1.plot(T_plot, u_pred,'b--',linewidth=2, markersize=0.5, label = 'Normalized PINNS')
ax1.plot(t, u_pred_un,'k-',linewidth=2, markersize=0.5, label = 'Unnormalized PINNS')
ax1.set_xlabel('$t$',fontsize=10)
ax1.set_ylabel('$u(x = 0.08)$',fontsize=10)

ax2.plot(t, p_test/133.33,'r-',linewidth=2, markersize=0.5, label = 'Discontinuous Galerkin')
ax2.plot(T_plot, p_pred/133.33,'b--',linewidth=2, markersize=0.5, label = 'Normalized PINNS')
ax2.plot(t, p_pred_un,'k-',linewidth=2, markersize=0.5, label = 'Unnormalized PINNS')
ax2.set_xlabel('$t$',fontsize=10)
ax2.set_ylabel('$p(x = 0.08)$',fontsize=10)
ax1.legend(loc='upper right', frameon=False, prop={'size': 10})
ax2.legend(loc='upper right', frameon=False, prop={'size': 10})

ax1.set_ylim(-0.5,5)   
ax2.set_ylim(0,1000)   


filename1 = "comparative_velocity_all.eps"
filename2 = "comparative_pressure_all.eps"

fig1.savefig(filename1, format = 'eps')
fig2.savefig(filename2, format = 'eps')    

plt.show()
