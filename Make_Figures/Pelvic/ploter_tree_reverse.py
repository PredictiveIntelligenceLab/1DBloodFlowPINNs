# -*- coding: utf-8 -*-
import plotting
import matplotlib.pyplot as plt
import numpy as np

results = np.load("results_reverse.npy").item()
test = np.load("test_data.npy").item()
test2 = np.load("vessel_6.npy").item()
test3 = np.load("vessel_2.npy").item()

T_f = results["Time_t"]
t = results["Time_t"]
T_plot = t
p_pred1= results["Pressure_vessel_3"]/133.3
p_pred2= results["Pressure_vessel_6"]/133.3
p_pred3= results["Pressure_vessel_2"]/133.3
p_pred4= results["Pressure_vessel_7"]/133.3
p_pred5= results["Pressure_vessel_5"]/133.3
p_pred6= results["Pressure_vessel_1"]/133.3
p_pred7= results["Pressure_vessel_4"]/133.3


u_pred1= results["Velocity_vessel_3"]
u_pred2= results["Velocity_vessel_6"]
u_pred3= results["Velocity_vessel_2"]
u_pred4= results["Velocity_vessel_7"]
u_pred5= results["Velocity_vessel_5"]
u_pred6= results["Velocity_vessel_1"]
u_pred7= results["Velocity_vessel_4"]

p_test1 = test["TestPress3"]/133.3
p_test2 = test["TestPress6"]/133.3
p_test3 = test["TestPress2"]/133.3
p_test4 = test["TestPress7"]/133.3
p_test5 = test["TestPress5"]/133.3
p_test6 = test["TestPress1"]/133.3
p_test7 = test["TestPress4"]/133.3


u_test1 = test["TestVel3"]
u_test2 = test["TestVel6"]
u_test3 = test["TestVel2"]
u_test4 = test["TestVel7"]
u_test5 = test["TestVel5"]
u_test6 = test["TestVel1"]
u_test7 = test["TestVel4"]



vessel_1 = np.load("bif_pred_3.npy").item()
vessel_2 = np.load("bif_pred_6.npy").item()
vessel_3 = np.load("bif_pred_2.npy").item()
vessel_4 = np.load("bif_pred_7.npy").item()
vessel_5 = np.load("bif_pred_5.npy").item()
vessel_6 = np.load("bif_pred_1.npy").item()
vessel_7 = np.load("bif_pred_4.npy").item()
 
u_b1 = vessel_1["Velocity"].T
u_b2 = vessel_2["Velocity"].T
u_b3 = vessel_3["Velocity"].T
u_b4 = vessel_4["Velocity"].T
u_b5 = vessel_5["Velocity"].T
u_b6 = vessel_6["Velocity"].T
u_b7 = vessel_7["Velocity"].T

p_b1 = vessel_1["Pressure"].T/133.3
p_b2 = vessel_2["Pressure"].T/133.3
p_b3 = vessel_3["Pressure"].T/133.3
p_b4 = vessel_4["Pressure"].T/133.3
p_b5 = vessel_5["Pressure"].T/133.3
p_b6 = vessel_6["Pressure"].T/133.3
p_b7 = vessel_6["Pressure"].T/133.3

fig1 = plt.figure(1,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig2 = plt.figure(2,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig3 = plt.figure(3,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig4 = plt.figure(4,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig5 = plt.figure(5,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig6 = plt.figure(6,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig7 = plt.figure(7,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig8 = plt.figure(8,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig9 = plt.figure(9,figsize=(15, 5), dpi=500, facecolor='w', frameon = False)
fig10 = plt.figure(10,figsize=(15,5), dpi=500, facecolor='w', frameon = False)
fig11 = plt.figure(11,figsize=(15,5), dpi=500, facecolor='w', frameon = False)
fig12 = plt.figure(12,figsize=(15,5), dpi=500, facecolor='w', frameon = False)
fig13 = plt.figure(13,figsize=(15,5), dpi=500, facecolor='w', frameon = False)
fig14 = plt.figure(14,figsize=(15,5), dpi=500, facecolor='w', frameon = False)
                                  
fig1.clf()                        
fig2.clf()                        
fig3.clf()                        
fig4.clf()
fig5.clf()
fig6.clf()
fig7.clf()
fig8.clf()
fig9.clf()
fig10.clf()
fig11.clf()
fig12.clf()
fig13.clf()
fig14.clf()

ax1 = fig1.add_subplot(111)  
ax2 = fig2.add_subplot(111)  
ax3 = fig3.add_subplot(111)  
ax4 = fig4.add_subplot(111)   
ax5 = fig5.add_subplot(111)   
ax6 = fig6.add_subplot(111)  
ax7 = fig7.add_subplot(111)  
ax8 = fig8.add_subplot(111)  
ax9 = fig9.add_subplot(111)  
ax10 = fig10.add_subplot(111)   
ax11 = fig11.add_subplot(111)   
ax12 = fig12.add_subplot(111)  
ax13 = fig13.add_subplot(111)   
ax14 = fig14.add_subplot(111)  

###############################################################################
ax1.plot(t, u_test1,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax1.plot(t, u_b1,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')

ax1.plot(T_f, u_pred1,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax1.set_xlabel('$t[s]$',fontsize=10)
ax1.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=10)

ax2.plot(t, u_test2,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax2.plot(t, u_b2,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')

ax2.plot(T_plot, u_pred2,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax2.set_xlabel('$t[s]$',fontsize=10)
ax2.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)

ax3.plot(t, u_test3,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax3.plot(t, u_b3,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')

ax3.plot(T_plot, u_pred3,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax3.set_xlabel('$t[s]$',fontsize=10)
ax3.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax3.tick_params(axis='both', which='major', labelsize=10)

ax4.plot(t, u_test4,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax4.plot(t, u_b4,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')

ax4.plot(T_f, u_pred4,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax4.set_xlabel('$t[s]$',fontsize=10)
ax4.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax4.tick_params(axis='both', which='major', labelsize=10)
ax5.plot(t, u_test5,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax5.plot(t, u_b5,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')
    
ax5.plot(T_plot, u_pred5,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax5.set_xlabel('$t[s]$',fontsize=10)
ax5.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax5.tick_params(axis='both', which='major', labelsize=10)

ax6.plot(t, u_test6,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax6.plot(t, u_b6,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')
ax6.plot(T_plot, u_pred6,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax6.set_xlabel('$t[s]$',fontsize=10)
ax6.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax6.tick_params(axis='both', which='major', labelsize=10)

ax7.plot(t, u_test7,'r-',linewidth=2, markersize=0.5, label = 'Reference Velocity')

ax7.plot(t, u_b7,'k.',linewidth=2, markersize=1.5, label='Predicted Velocity (DG with discovered RCR)')
ax7.plot(T_plot, u_pred7,'b--',linewidth=2, markersize=0.5, label='Predicted Velocity PINNS')

ax7.set_xlabel('$t[s]$',fontsize=10)
ax7.set_ylabel('$u(t)[m/s]$',fontsize=10)
ax7.tick_params(axis='both', which='major', labelsize=10)

###############################################################################

ax8.plot(t, p_test1,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')
ax8.plot(t, p_b1,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax8.plot(T_plot, p_pred1,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax8.set_xlabel('$t[s]$',fontsize=10)
ax8.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax8.tick_params(axis='both', which='major', labelsize=10)
ax9.plot(t, p_test2,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')

ax9.plot(t, p_b2,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax9.plot(T_plot, p_pred2,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax9.set_xlabel('$t[s]$',fontsize=10)
ax9.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax9.tick_params(axis='both', which='major', labelsize=10)
ax10.plot(t, p_test3,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')

ax10.plot(t, p_b3,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax10.plot(T_plot, p_pred3,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax10.set_xlabel('$t[s]$',fontsize=10)
ax10.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax10.tick_params(axis='both', which='major', labelsize=10)
ax11.plot(t, p_test4,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')

ax11.plot(t, p_b4,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax11.plot(T_plot, p_pred4,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax11.set_xlabel('$t[s]$',fontsize=10)
ax11.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax11.tick_params(axis='both', which='major', labelsize=10)
ax12.plot(t, p_test5,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')

ax12.plot(t, p_b5,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax12.plot(T_plot, p_pred5,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax12.set_xlabel('$t[s]$',fontsize=10)
ax12.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax13.plot(t, p_test6,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')

ax13.plot(t, p_b6,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax13.plot(T_plot, p_pred6,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax13.set_xlabel('$t[s]$',fontsize=10)
ax13.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax13.tick_params(axis='both', which='major', labelsize=10)
ax14.plot(t, p_test7,'r-',linewidth=2, markersize=0.5, label = 'Reference Pressure')

ax14.plot(t, p_b7,'k.',linewidth=2, markersize=1.5, label='Predicted Pressure (DG with discovered RCR)')

ax14.plot(T_plot, p_pred7,'b--',linewidth=2, markersize=0.5, label='Predicted Pressure PINNS')

ax14.set_xlabel('$t[s]$',fontsize=10)
ax14.set_ylabel('$p(t)[mmHg]$',fontsize=10)
ax14.tick_params(axis='both', which='major', labelsize=10)


ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
ax7.legend()
ax8.legend()
ax9.legend()
ax10.legend()
ax11.legend()
ax12.legend()
ax13.legend()
ax14.legend()

ax1.set_ylim(-0.55,4.9)   
ax2.set_ylim(-0.38, 2.7)
ax3.set_ylim(-0.3, 2.7)
ax4.set_ylim(-0.25,1.5)   
ax5.set_ylim(-0.15,1.5)
ax6.set_ylim(-0.20,1.6)
ax7.set_ylim(-0.15,1.6)   
ax8.set_ylim(0,220)   
ax9.set_ylim(0,220)
ax10.set_ylim(0,220)
ax11.set_ylim(0,220)   
ax12.set_ylim(0,220)
ax13.set_ylim(0,220)
ax14.set_ylim(0,220)

filename1 = "comparative_velocity1_tree_RCR.png"
filename2 = "comparative_velocity2_tree_RCR.png"
filename3 = "comparative_velocity3_tree_RCR.png"
filename4 = "comparative_velocity4_tree_RCR.png"
filename5 = "comparative_velocity5_tree_RCR.png"
filename6 = "comparative_velocity6_tree_RCR.png"
filename7 = "comparative_velocity7_tree_RCR.png"
filename8 = "comparative_pressure1_tree_RCR.png"
filename9 = "comparative_pressure2_tree_RCR.png"
filename10 = "comparative_pressure3_tree_RCR.png"
filename11 = "comparative_pressure4_tree_RCR.png"
filename12 = "comparative_pressure5_tree_RCR.png"
filename13 = "comparative_pressure6_tree_RCR.png"
filename14 = "comparative_pressure7_tree_RCR.png"

fig1.savefig(filename1, format = 'png')
fig2.savefig(filename2, format = 'png')    
fig3.savefig(filename3, format = 'png')
fig4.savefig(filename4, format = 'png')    
fig5.savefig(filename5, format = 'png')
fig6.savefig(filename6, format = 'png')    
fig7.savefig(filename7, format = 'png')
fig8.savefig(filename8, format = 'png')    
fig9.savefig(filename9, format = 'png')
fig10.savefig(filename10, format = 'png')    
fig11.savefig(filename11, format = 'png')
fig12.savefig(filename12, format = 'png')    
fig13.savefig(filename13, format = 'png')
fig14.savefig(filename14, format = 'png')    

plt.show()
