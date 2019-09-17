import matplotlib.pyplot as plt
import numpy as np
from pinn_ns_bif import PhysicsInformedNN

def get_equilibrium_cross_sectional_area_aorta_1(x):
    X1 = 0.0
    X2 = 0.04964
    denom = X2-X1
    x1 = 2.293820e-04
    x2 = 2.636589e-04
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

def get_equilibrium_cross_sectional_area_carotid(x):
    X1 = 0.04964
    X2 = 0.10284
    denom = X2-X1
    x1 = 2.636589e-04
    x2 = 2.623127e-05
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

def get_equilibrium_cross_sectional_area_aorta_3(x):
    X1 = 0.04964
    X2 = 0.1383
    denom = X2-X1
    x1 = 2.636589e-04
    x2 = 2.177177e-04
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

def get_equilibrium_cross_sectional_area_aorta_4(x):
    X1 = 0.1383
    X2 = 0.17056
    denom = X2-X1
    x1 = 2.177177e-04
    x2 = 2.411245e-04
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

if __name__ == "__main__":
    
    # Define the number of spatio-temporal domain points to evaluate the residual
    # of the system of equations.
    
    N_f =  2000
    
    aorta1_velocity = np.load("Aorta1_U.npy").item()
    aorta2_velocity = np.load("Aorta2_U.npy").item()
    aorta4_velocity = np.load("Aorta4_U.npy").item()
    carotid_velocity= np.load("LCommonCarotid_U.npy").item()

    aorta1_area = np.load("Aorta1_A.npy").item()
    aorta2_area = np.load("Aorta2_A.npy").item()
    aorta4_area = np.load("Aorta4_A.npy").item()
    carotid_area = np.load("LCommonCarotid_A.npy").item()
    
    test_aorta3_velocity = np.load("Aorta3_U.npy").item()
    test_aorta3_area = np.load("Aorta3_A.npy").item()
    
    t = aorta1_velocity['t']*1e-3
    
    velocity_measurements_aorta1 = aorta1_velocity["U"]*1e-2
    velocity_measurements_carotid = carotid_velocity["U"]*1e-2
    velocity_measurements_aorta4 = aorta4_velocity["U"]*1e-2
    
    area_measurements_aorta1 = aorta1_area["A"]*1e-6
    area_measurements_carotid = carotid_area["A"]*1e-6
    area_measurements_aorta4 = aorta4_area["A"]*1e-6

    velocity_testpoint_aorta3 = test_aorta3_velocity["U"]*1e-2
    area_testpoint_aorta3 = test_aorta3_area["A"]*1e-6
    
    u_test1 = aorta2_velocity['U']*1e-2
    A_test1 = aorta2_area['A']*1e-6
    
    # Number of measurements
    
    N_u = t.shape[0]

    layers = [2, 100, 100, 100, 100, 100, 100, 3]
    
    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)
    
    lower_bound_vessel_1 = 0.0   
    upper_bound_vessel_1 = 0.04964
    
    lower_bound_vessel_2 = 0.04964
    upper_bound_vessel_2 = 0.10284
    
    lower_bound_vessel_3 = 0.04964
    upper_bound_vessel_3 = 0.1383

    lower_bound_vessel_4 = 0.1383
    upper_bound_vessel_4 = 0.17056
    
    # Spatial/temporal coordinates for initial conditions
    X_initial_aorta1 = np.linspace(lower_bound_vessel_1,upper_bound_vessel_1,N_u)[:,None]
    X_initial_carotid = np.linspace(lower_bound_vessel_2,upper_bound_vessel_2,N_u)[:,None]
    X_initial_aorta3 = np.linspace(lower_bound_vessel_3,upper_bound_vessel_3,N_u)[:,None]
    X_initial_aorta4 = np.linspace(lower_bound_vessel_4,upper_bound_vessel_4,N_u)[:,None]
    
    T_initial  = lower_bound_t*np.ones((N_u))[:,None]
    
    # Spatial/temporal coordinates for boundary conditions
    X_boundary_aorta1 = lower_bound_vessel_1*np.ones((N_u))[:,None]
    X_boundary_carotid = upper_bound_vessel_2*np.ones((N_u))[:,None]
    X_boundary_aorta3 = upper_bound_vessel_3*np.ones((N_u))[:,None]
    X_boundary_aorta4 = upper_bound_vessel_4*np.ones((N_u))[:,None]

    T_boundary = t
    
    # Measurement Spatial/temporal coordinates
    X_measurement_aorta1 = np.vstack((X_initial_aorta1, X_boundary_aorta1))
    X_measurement_carotid = np.vstack((X_initial_carotid, X_boundary_carotid))    
    X_measurement_aorta3 = np.vstack((X_initial_aorta3))    
    X_measurement_aorta4 = np.vstack((X_initial_aorta4, X_boundary_aorta4))    
    
    T_measurement = np.vstack((T_initial, T_boundary))

    X_residual_aorta1 = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1)*np.random.random((N_f))[:,None]
    X_residual_carotid = lower_bound_vessel_2 + (upper_bound_vessel_2-lower_bound_vessel_2)*np.random.random((N_f))[:,None]
    X_residual_aorta3 = lower_bound_vessel_3 + (upper_bound_vessel_3-lower_bound_vessel_3)*np.random.random((N_f))[:,None]
    X_residual_aorta4 = lower_bound_vessel_4 + (upper_bound_vessel_4-lower_bound_vessel_4)*np.random.random((N_f))[:,None]
    
    T_residual = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]
        
    A_initial_aorta1 = get_equilibrium_cross_sectional_area_aorta_1(X_initial_aorta1)
    A_initial_carotid = get_equilibrium_cross_sectional_area_carotid(X_initial_carotid)
    A_initial_aorta3 = get_equilibrium_cross_sectional_area_aorta_3(X_initial_aorta3)    
    A_initial_aorta4 = get_equilibrium_cross_sectional_area_aorta_4(X_initial_aorta4)
    
    
    U_initial_aorta1 = velocity_measurements_aorta1[0]*np.ones((N_u,1))
    U_initial_aorta2 = velocity_measurements_carotid[0]*np.ones((N_u,1))
    U_initial_aorta3 = velocity_testpoint_aorta3[0]*np.ones((N_u,1))
    U_initial_aorta4 = velocity_measurements_aorta4[0]*np.ones((N_u,1))
         
    A_training_aorta1 = np.vstack((A_initial_aorta1,area_measurements_aorta1))
    U_training_aorta1 = np.vstack((U_initial_aorta1,velocity_measurements_aorta1))

    A_training_carotid = np.vstack((A_initial_carotid,area_measurements_carotid))
    U_training_carotid = np.vstack((U_initial_aorta2,velocity_measurements_carotid))
    
    A_training_aorta3 = np.vstack((A_initial_aorta3))
    U_training_aorta3 = np.vstack((U_initial_aorta3))

    A_training_aorta4 = np.vstack((A_initial_aorta4,area_measurements_aorta4))
    U_training_aorta4 = np.vstack((U_initial_aorta4,velocity_measurements_aorta4))
    
    bif_points = [upper_bound_vessel_1, upper_bound_vessel_3]
    
    model = PhysicsInformedNN(X_measurement_aorta1, X_measurement_carotid,\
                              X_measurement_aorta3, X_measurement_aorta4,\
                              T_measurement, T_initial, 
                              A_training_aorta1,  U_training_aorta1,\
                              A_training_carotid, U_training_carotid,\
                              A_training_aorta3,  U_training_aorta3,\
                              A_training_aorta4,  U_training_aorta4, \
                              X_residual_aorta1, \
                              X_residual_carotid, \
                              X_residual_aorta3, \
                              X_residual_aorta4,\
                              T_residual,layers,bif_points)
    
    model.train(70000,1e-3)
    model.train(35000,1e-4)
     
    test_point1 = 0.04964*np.ones((X_residual_aorta1.shape[0],1))    
    test_point3 = 0.1383*np.ones((t.shape[0],1))
        
    test_aorta1_lboundary = lower_bound_vessel_1*np.ones((t.shape[0],1))
    test_carotid_lboundary = lower_bound_vessel_2*np.ones((t.shape[0],1))
    test_aorta4_lboundary = lower_bound_vessel_4*np.ones((t.shape[0],1))
    
    A_predict_aorta1, u_predict_aorta1, p_predict_aorta1     = model.predict_aorta1(test_point1, T_residual)
    A_predict_carotid, u_predict_carotid, p_predict_carotid    = model.predict_carotid(test_point1, T_residual)
    A_predict_aorta3l, u_predict_aorta3l, p_predict_aorta3l  = model.predict_aorta3(test_point1, T_residual)
    A_predict_aorta4, u_predict_aorta4, p_predict_aorta4  = model.predict_aorta4(test_point3, t)
    
    A_pred1b, u_pred1b, p_pred1b  = model.predict_aorta1(test_aorta1_lboundary, t)
    A_pred2b, u_pred2b, p_pred2b  = model.predict_carotid(test_carotid_lboundary, t)
    A_pred3b, u_pred3b, p_pred3b  = model.predict_aorta4(test_aorta4_lboundary, t)

    fig1 = plt.figure(1,figsize=(10, 6), dpi=400, facecolor='w', frameon = False)
    fig2 = plt.figure(2,figsize=(10, 6), dpi=400, facecolor='w', frameon = False)
    fig3 = plt.figure(3,figsize=(10, 6), dpi=400, facecolor='w', frameon = False)

    fig1.clf()
    fig2.clf()
    fig3.clf()
    
    ax1 = fig1.add_subplot(111)  
    ax2 = fig2.add_subplot(111)  
    ax3 = fig3.add_subplot(111)  
   
    ax1.plot(t, u_predict_aorta4,'r--',linewidth=3.5, markersize=2.5)
    ax1.plot(t, velocity_testpoint_aorta3,'b-',linewidth=3.5, markersize=2.5)

    ax1.set_xlabel('Time in $s$')
    ax1.set_ylabel('Velocity in $m/s$')
    ax1.set_title('Compare velocity aorta3')
    
    ax2.plot(t, A_predict_aorta4,'r--',linewidth=3.5, markersize=2.5)
    ax2.plot(t, area_testpoint_aorta3,'b-',linewidth=3.5, markersize=2.5)

    ax2.set_xlabel('Time in $s$')
    ax2.set_ylabel('Area in $mm^2$')
    ax2.set_title('Compare area aorta3')
    
    ax3.plot(t, p_predict_aorta4/133.,'r--',linewidth=3.5, markersize=2.5)

    ax3.set_xlabel('Time in $s$')
    ax3.set_ylabel('Pressure in $mmHg$')
    ax3.set_title('Pressure aorta3')
    
