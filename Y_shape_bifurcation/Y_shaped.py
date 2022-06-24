import matplotlib.pyplot as plt
import numpy as np

from Y_shaped_pinns import OneDBioPINN

if __name__ == "__main__":
    N_f =  2000
    
    input_vessel_1 = np.load("input_1.npy").item()
    output_vessel_2 = np.load("output_2.npy").item()
    output_vessel_3 = np.load("output_3.npy").item()
    
    t = input_vessel_1["Time"][:,None]
    
    test_vessel_1 = np.load("test_1.npy").item()
    test_vessel_2 = np.load("test_2.npy").item()
    test_vessel_3 = np.load("test_3.npy").item()

    velocity_measurements_vessel1 = input_vessel_1["Velocity"][:,None]
    velocity_measurements_vessel2 = output_vessel_2["Velocity"][:,None]
    velocity_measurements_vessel3 = output_vessel_3["Velocity"][:,None]
    
    area_measurements_vessel1 = input_vessel_1["Area"][:,None]
    area_measurements_vessel2 = output_vessel_2["Area"][:,None]
    area_measurements_vessel3 = output_vessel_3["Area"][:,None]
        
    velocity_test_vessel1 = test_vessel_1["Velocity"][:,None]
    velocity_test_vessel2 = test_vessel_2["Velocity"][:,None]
    velocity_test_vessel3 = test_vessel_3["Velocity"][:,None]

    pressure_test_vessel1 = test_vessel_1["Pressure"][:,None]
    pressure_test_vessel2 = test_vessel_2["Pressure"][:,None]
    pressure_test_vessel3 = test_vessel_3["Pressure"][:,None]
    
    N_u = t.shape[0]
    
    layers = [2, 100, 100, 100, 100, 100, 100, 100, 3]   
    
    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)
    
    lower_bound_vessel_1 = 0.0   
    upper_bound_vessel_1 = 0.1703
    
    lower_bound_vessel_2 = 0.1703
    upper_bound_vessel_2 = 0.1773
    
    lower_bound_vessel_3 = 0.1703
    upper_bound_vessel_3 = 0.1770
    
    bif_points = 0.1703
    
    X_initial_vessel1 = np.linspace(lower_bound_vessel_1,upper_bound_vessel_1,N_u)[:,None]
    X_initial_vessel2 = np.linspace(lower_bound_vessel_2,upper_bound_vessel_2,N_u)[:,None]
    X_initial_vessel3 = np.linspace(lower_bound_vessel_3,upper_bound_vessel_3,N_u)[:,None]
    
    T_initial  = lower_bound_t*np.ones((N_u))[:,None]
    
    X_boundary_vessel1 = lower_bound_vessel_1*np.ones((N_u))[:,None]
    X_boundary_vessel2 = upper_bound_vessel_2*np.ones((N_u))[:,None]
    X_boundary_vessel3 = upper_bound_vessel_3*np.ones((N_u))[:,None]

    T_boundary = t
        
    X_measurement_vessel1 = np.vstack((X_initial_vessel1, X_boundary_vessel1))
    X_measurement_vessel2 = np.vstack((X_initial_vessel2, X_boundary_vessel2))    
    X_measurement_vessel3 = np.vstack((X_initial_vessel3, X_boundary_vessel3))    
    
    T_measurement = np.vstack((T_initial, T_boundary))

    X_residual_vessel1 = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1)*np.random.random((N_f))[:,None]
    X_residual_vessel2 = lower_bound_vessel_2 + (upper_bound_vessel_2-lower_bound_vessel_2)*np.random.random((N_f))[:,None]
    X_residual_vessel3 = lower_bound_vessel_3 + (upper_bound_vessel_3-lower_bound_vessel_3)*np.random.random((N_f))[:,None]
    
    T_residual = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]
   
    A_initial_vessel1 = 1.35676200E-05*np.ones((N_u,1))
    A_initial_vessel2 = 1.81458400E-06*np.ones((N_u,1))
    A_initial_vessel3 = 1.35676200E-05*np.ones((N_u,1))

    U_initial_vessel1 = 0.*np.ones((N_u,1))
    U_initial_vessel2 = 0.*np.ones((N_u,1))
    U_initial_vessel3 = 0.*np.ones((N_u,1))
        
    A_training_vessel1 = np.vstack((A_initial_vessel1,area_measurements_vessel1))
    U_training_vessel1 = np.vstack((U_initial_vessel1,velocity_measurements_vessel1))

    A_training_vessel2 = np.vstack((A_initial_vessel2,area_measurements_vessel2))
    U_training_vessel2 = np.vstack((U_initial_vessel2,velocity_measurements_vessel2))
    
    A_training_vessel3 = np.vstack((A_initial_vessel3,area_measurements_vessel3))
    U_training_vessel3 = np.vstack((U_initial_vessel3,velocity_measurements_vessel3))
              
    model = OneDBioPINN(X_measurement_vessel1, 
                               X_measurement_vessel2,
                               X_measurement_vessel3,
                               A_training_vessel1,  U_training_vessel1,
                               A_training_vessel2,  U_training_vessel2,
                               A_training_vessel3,  U_training_vessel3,
                               X_residual_vessel1, 
                               X_residual_vessel2, 
                               X_residual_vessel3, 
                               T_residual, T_measurement, layers, bif_points)
    model.train(90000,1e-3)
    model.train(40000,1e-4)
            
    X_test_vessel1 = 0.1*np.ones((T_residual.shape[0],1))
    X_test_vessel2 = 0.176*np.ones((T_residual.shape[0],1))
    X_test_vessel3 = 0.174*np.ones((T_residual.shape[0],1))
        
    A_predicted_vessel1, U_predicted_vessel1, p_predicted_vessel1  = model.predict_vessel1(X_test_vessel1, T_residual)
    A_predicted_vessel2, U_predicted_vessel2, p_predicted_vessel2  = model.predict_vessel2(X_test_vessel2, T_residual)
    A_predicted_vessel3, U_predicted_vessel3, p_predicted_vessel3  = model.predict_vessel3(X_test_vessel3, T_residual)
    
    fig1 = plt.figure(1,figsize=(22, 22), dpi=300, facecolor='w', frameon = False)
    fig2 = plt.figure(2,figsize=(22, 22), dpi=300, facecolor='w', frameon = False)
    
    ax11 = fig1.add_subplot(131)  
    ax12 = fig1.add_subplot(132)  
    ax13 = fig1.add_subplot(133)  
    
    ax21 = fig2.add_subplot(131)  
    ax22 = fig2.add_subplot(132)  
    ax23 = fig2.add_subplot(133)  

    ax11.plot(T_residual, U_predicted_vessel1,'r-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')
    ax11.plot(t, velocity_test_vessel1,'b--',linewidth=1, markersize=0.5, label='Reference velocity Vessel1')

    ax12.plot(T_residual, U_predicted_vessel2,'r-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel2')
    ax12.plot(t, velocity_test_vessel2,'b--',linewidth=1, markersize=0.5, label='Reference velocity Vessel2')

    ax13.plot(T_residual, U_predicted_vessel3,'r-',linewidth=1, markersize=0.5, label='Predicted velocity Vessel3')
    ax13.plot(t, velocity_test_vessel3,'b--',linewidth=1, markersize=0.5, label='Reference velocity Vessel3')

    fig1.suptitle('Comparative velocity')
    ax11.set_xlabel("t in s")
    ax11.set_ylabel("Velocity in m/s")
    ax12.set_xlabel("t in s")
    ax12.set_ylabel("Velocity in m/s")
    ax13.set_xlabel("t in s")
    ax13.set_ylabel("Velocity in m/s")

    ax21.plot(T_residual, p_predicted_vessel1,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
    ax21.plot(t, pressure_test_vessel1,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel1')

    ax22.plot(T_residual, p_predicted_vessel2,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel2')
    ax22.plot(t, pressure_test_vessel2,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel2')

    ax23.plot(T_residual, p_predicted_vessel3,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel3')
    ax23.plot(t, pressure_test_vessel3,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel3')


    fig2.suptitle('Comparative pressure')
    ax21.set_xlabel("t in s")
    ax21.set_ylabel("Pressure in Pa")
    ax22.set_xlabel("t in s")
    ax22.set_ylabel("Pressure in Pa")
    ax23.set_xlabel("t in s")
    ax23.set_ylabel("Pressure in Pa")
    
    fig1.savefig("Comparative_Velocity.png")
    fig2.savefig("Comparative_Pressure.png")
    
