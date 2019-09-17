import plotting 
import matplotlib.pyplot as plt
import numpy as np

results = np.load("Losses.npy").item()

total_loss = results['Total_Loss'][:,None]
loss_area  = results['loss_area' ][:,None]
loss_velo  = results['loss_velo' ][:,None]
loss_res   = results['loss_res'  ][:,None]
loss_press = results['loss_press'][:,None]
loss_cont  = results['loss_cont' ][:,None]

iteration = np.arange(0,total_loss.shape[0])*100

fig1 = plt.figure(1,figsize=(12, 7), dpi=300, facecolor='w', frameon = False)

fig1.clf()

ax1 = fig1.add_subplot(111)  

ax1.plot(iteration, loss_cont,'m-',linewidth=2, markersize=0.5, alpha = 0.8, label = 'Continuity Loss' )
ax1.plot(iteration, loss_area,'b-',linewidth=2, markersize=0.5, alpha = 0.8, label = 'Reconstruction Loss for area' )
ax1.plot(iteration, loss_velo,'r-',linewidth=2, markersize=0.5, alpha = 0.8, label = 'Reconstruction Loss for velocity' )
ax1.plot(iteration, loss_res,'k-',linewidth=2, markersize=0.5, alpha = 0.8, label = 'Residual Loss'  )
ax1.plot(iteration, loss_press,'g-',linewidth=2, markersize=0.5, alpha = 0.8, label = 'Pressure Loss' )

ax1.legend()
ax1.set_yscale('log')
filename1 = "losses.png"
#
fig1.savefig(filename1, format = 'png')

plt.show()
