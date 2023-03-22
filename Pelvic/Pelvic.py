import matplotlib.pyplot as plt
import numpy as np

from pelvic_pinns import OneDBioPINN
                
if __name__ == "__main__":
    
    
    N_f =  2000
    
    # We indicate the 
    test_vessel_1 = np.load("vessel_1.npy", allow_pickle=True).item()
    test_vessel_2 = np.load("vessel_2.npy", allow_pickle=True).item()
    test_vessel_3 = np.load("vessel_3.npy", allow_pickle=True).item()
    test_vessel_4 = np.load("vessel_4.npy", allow_pickle=True).item()
    test_vessel_5 = np.load("vessel_5.npy", allow_pickle=True).item()
    test_vessel_6 = np.load("vessel_6.npy", allow_pickle=True).item()
    test_vessel_7 = np.load("vessel_7.npy", allow_pickle=True).item()
    
    output_vessel_1 = np.load("test_vessel_1.npy", allow_pickle=True).item()
    vessel_2        = np.load("test_vessel_2.npy", allow_pickle=True).item()
    input_vessel_3  = np.load("test_vessel_3.npy", allow_pickle=True).item()
    output_vessel_4 = np.load("test_vessel_4.npy", allow_pickle=True).item()
    output_vessel_5 = np.load("test_vessel_5.npy", allow_pickle=True).item()
    vessel_6        = np.load("test_vessel_6.npy", allow_pickle=True).item()
    output_vessel_7 = np.load("test_vessel_7.npy", allow_pickle=True).item()

    velocity_test_vessel1 = test_vessel_1["Velocity"][:,None]
    velocity_test_vessel2 = test_vessel_2["Velocity"][:,None]
    velocity_test_vessel3 = test_vessel_3["Velocity"][:,None]
    velocity_test_vessel4 = test_vessel_4["Velocity"][:,None]
    velocity_test_vessel5 = test_vessel_5["Velocity"][:,None]
    velocity_test_vessel6 = test_vessel_6["Velocity"][:,None]
    velocity_test_vessel7 = test_vessel_7["Velocity"][:,None]

    pressure_test_vessel1 = test_vessel_1["Pressure"][:,None]
    pressure_test_vessel2 = test_vessel_2["Pressure"][:,None]
    pressure_test_vessel3 = test_vessel_3["Pressure"][:,None]
    pressure_test_vessel4 = test_vessel_4["Pressure"][:,None]
    pressure_test_vessel5 = test_vessel_5["Pressure"][:,None]
    pressure_test_vessel6 = test_vessel_6["Pressure"][:,None]
    pressure_test_vessel7 = test_vessel_7["Pressure"][:,None]

    t = input_vessel_3["Time"][:,None]
    t = t - t.min(0)

    velocity_measurements_vessel1 = output_vessel_1["Velocity"][:,None]
    velocity_measurements_vessel2 = vessel_2["Velocity"][:,None]
    velocity_measurements_vessel3 = input_vessel_3["Velocity"][:,None]
    velocity_measurements_vessel4 = output_vessel_4["Velocity"][:,None]
    velocity_measurements_vessel5 = output_vessel_5["Velocity"][:,None]
    velocity_measurements_vessel6 = vessel_6["Velocity"][:,None]
    velocity_measurements_vessel7 = output_vessel_7["Velocity"][:,None]
   
    area_measurements_vessel1 = output_vessel_1["Area"][:,None]
    area_measurements_vessel2 = vessel_2["Area"][:,None]
    area_measurements_vessel3 = input_vessel_3["Area"][:,None]
    area_measurements_vessel4 = output_vessel_4["Area"][:,None]
    area_measurements_vessel5 = output_vessel_5["Area"][:,None]
    area_measurements_vessel6 = vessel_6["Area"][:,None]   
    area_measurements_vessel7 = output_vessel_7["Area"][:,None]
   
    N_u = t.shape[0]
    
    
    layers = [2, 100, 100, 100, 100, 100, 100, 100, 3]
    
    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)
    
    lower_bound_vessel_3 = 0.0      
    upper_bound_vessel_3 =0.01068202
    
    lower_bound_vessel_6 = upper_bound_vessel_3
    upper_bound_vessel_6 = upper_bound_vessel_3 + 0.06666379

    lower_bound_vessel_2 = upper_bound_vessel_3
    upper_bound_vessel_2 = upper_bound_vessel_3 + 0.0699352

    lower_bound_vessel_4 = upper_bound_vessel_2   
    upper_bound_vessel_4 = upper_bound_vessel_2 + 0.13438403
    
    lower_bound_vessel_1 = upper_bound_vessel_2 
    upper_bound_vessel_1 = upper_bound_vessel_2 + 0.13642118
       
    lower_bound_vessel_5 = upper_bound_vessel_6
    upper_bound_vessel_5 = upper_bound_vessel_6 + 0.1495032
    
    lower_bound_vessel_7 = upper_bound_vessel_6
    upper_bound_vessel_7 = upper_bound_vessel_6 + 0.14773513
    
    measurement_points = np.array([0.14882781, 0.04564962, 0.00534101, 0.14780923, 0.15209741,
       0.04401392, 0.15121337])   
    
    test_point = np.array([ upper_bound_vessel_1, 0.06, lower_bound_vessel_3, upper_bound_vessel_4, upper_bound_vessel_5,
       0.03, upper_bound_vessel_7])    
       
    bif_points = [upper_bound_vessel_3, upper_bound_vessel_6, upper_bound_vessel_2]
    
    X_initial_vessel1 = np.linspace(lower_bound_vessel_1,upper_bound_vessel_1,N_u)[:,None]
    X_initial_vessel2 = np.linspace(lower_bound_vessel_2,upper_bound_vessel_2,N_u)[:,None]
    X_initial_vessel3 = np.linspace(lower_bound_vessel_3,upper_bound_vessel_3,N_u)[:,None]
    X_initial_vessel4 = np.linspace(lower_bound_vessel_4,upper_bound_vessel_4,N_u)[:,None]
    X_initial_vessel5 = np.linspace(lower_bound_vessel_5,upper_bound_vessel_5,N_u)[:,None]
    X_initial_vessel6 = np.linspace(lower_bound_vessel_6,upper_bound_vessel_6,N_u)[:,None]
    X_initial_vessel7 = np.linspace(lower_bound_vessel_7,upper_bound_vessel_7,N_u)[:,None]
    
    T_initial  = lower_bound_t*np.ones((N_u))[:,None]
    
    X_boundary_vessel3 = measurement_points[2]*np.ones((N_u))[:,None]
    X_boundary_vessel7 = measurement_points[6]*np.ones((N_u))[:,None]
    X_boundary_vessel5 = measurement_points[4]*np.ones((N_u))[:,None]
    X_boundary_vessel1 = measurement_points[0]*np.ones((N_u))[:,None]
    X_boundary_vessel4 = measurement_points[3]*np.ones((N_u))[:,None]

    T_boundary = t

    X_measurement_vessel1 = np.vstack((X_initial_vessel1, X_boundary_vessel1))    
    X_measurement_vessel2 = np.vstack((X_initial_vessel2))       
    X_measurement_vessel3 = np.vstack((X_initial_vessel3, X_boundary_vessel3))
    X_measurement_vessel4 = np.vstack((X_initial_vessel4, X_boundary_vessel4))    
    X_measurement_vessel5 = np.vstack((X_initial_vessel5, X_boundary_vessel5))    
    X_measurement_vessel6 = np.vstack((X_initial_vessel6))    
    X_measurement_vessel7 = np.vstack((X_initial_vessel7, X_boundary_vessel7))    
  
    
    T_measurement = np.vstack((T_initial, T_boundary))

    X_residual_vessel1 = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1)*np.random.random((N_f))[:,None]
    X_residual_vessel2 = lower_bound_vessel_2 + (upper_bound_vessel_2-lower_bound_vessel_2)*np.random.random((N_f))[:,None]
    X_residual_vessel3 = lower_bound_vessel_3 + (upper_bound_vessel_3-lower_bound_vessel_3)*np.random.random((N_f))[:,None]
    X_residual_vessel4 = lower_bound_vessel_4 + (upper_bound_vessel_4-lower_bound_vessel_4)*np.random.random((N_f))[:,None]
    X_residual_vessel5 = lower_bound_vessel_5 + (upper_bound_vessel_5-lower_bound_vessel_5)*np.random.random((N_f))[:,None]
    X_residual_vessel6 = lower_bound_vessel_6 + (upper_bound_vessel_6-lower_bound_vessel_6)*np.random.random((N_f))[:,None]
    X_residual_vessel7 = lower_bound_vessel_7 + (upper_bound_vessel_7-lower_bound_vessel_7)*np.random.random((N_f))[:,None]
    
    T_residual = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]
   
    A_initial_vessel1 = area_measurements_vessel1[0,0]*np.ones((N_u,1))
    A_initial_vessel2 = area_measurements_vessel2[0,0]*np.ones((N_u,1))
    A_initial_vessel3 = area_measurements_vessel3[0,0]*np.ones((N_u,1))
    A_initial_vessel4 = area_measurements_vessel4[0,0]*np.ones((N_u,1))
    A_initial_vessel5 = area_measurements_vessel5[0,0]*np.ones((N_u,1))
    A_initial_vessel6 = area_measurements_vessel6[0,0]*np.ones((N_u,1))
    A_initial_vessel7 = area_measurements_vessel7[0,0]*np.ones((N_u,1))

    U_initial_vessel1 = velocity_measurements_vessel1[0,0]*np.ones((N_u,1))
    U_initial_vessel2 = velocity_measurements_vessel2[0,0]*np.ones((N_u,1))
    U_initial_vessel3 = velocity_measurements_vessel3[0,0]*np.ones((N_u,1))
    U_initial_vessel4 = velocity_measurements_vessel4[0,0]*np.ones((N_u,1))
    U_initial_vessel5 = velocity_measurements_vessel5[0,0]*np.ones((N_u,1))
    U_initial_vessel6 = velocity_measurements_vessel6[0,0]*np.ones((N_u,1))
    U_initial_vessel7 = velocity_measurements_vessel7[0,0]*np.ones((N_u,1))
        
    A_training_vessel1 = np.vstack((A_initial_vessel1,area_measurements_vessel1))
    U_training_vessel1 = np.vstack((U_initial_vessel1,velocity_measurements_vessel1))

    A_training_vessel2 = np.vstack((A_initial_vessel2))
    U_training_vessel2 = np.vstack((U_initial_vessel2))
    
    A_training_vessel3 = np.vstack((A_initial_vessel3,area_measurements_vessel3))
    U_training_vessel3 = np.vstack((U_initial_vessel3,velocity_measurements_vessel3))
 
    A_training_vessel4 = np.vstack((A_initial_vessel4,area_measurements_vessel4))
    U_training_vessel4 = np.vstack((U_initial_vessel4,velocity_measurements_vessel4))

    A_training_vessel5 = np.vstack((A_initial_vessel5,area_measurements_vessel5))
    U_training_vessel5 = np.vstack((U_initial_vessel5,velocity_measurements_vessel5))
    
    A_training_vessel6 = np.vstack((A_initial_vessel6))
    U_training_vessel6 = np.vstack((U_initial_vessel6))

    A_training_vessel7 = np.vstack((A_initial_vessel7,area_measurements_vessel7))
    U_training_vessel7 = np.vstack((U_initial_vessel7,velocity_measurements_vessel7))
     
    model = OneDBioPINN(X_measurement_vessel1, 
                               X_measurement_vessel2,
                               X_measurement_vessel3,
                               X_measurement_vessel4,
                               X_measurement_vessel5,
                               X_measurement_vessel6,
                               X_measurement_vessel7,
                               A_training_vessel1,  U_training_vessel1,
                               A_training_vessel2,  U_training_vessel2,
                               A_training_vessel3,  U_training_vessel3,
                               A_training_vessel4,  U_training_vessel4, 
                               A_training_vessel5,  U_training_vessel5, 
                               A_training_vessel6,  U_training_vessel6, 
                               A_training_vessel7,  U_training_vessel7, 
                               X_residual_vessel1, 
                               X_residual_vessel2, 
                               X_residual_vessel3, 
                               X_residual_vessel4,
                               X_residual_vessel5,
                               X_residual_vessel6,
                               X_residual_vessel7,
                               T_residual,T_measurement, layers, bif_points, T_initial)


    model.train(290000,1e-3)
    model.train(50000,1e-4)

    X_test_vessel1 = test_point[0]*np.ones((X_residual_vessel1.shape[0],1))
    X_test_vessel2 = test_point[1]*np.ones((X_residual_vessel2.shape[0],1))
    X_test_vessel3 = test_point[2]*np.ones((X_residual_vessel3.shape[0],1))
    X_test_vessel4 = test_point[3]*np.ones((X_residual_vessel4.shape[0],1))
    X_test_vessel5 = test_point[4]*np.ones((X_residual_vessel5.shape[0],1))
    X_test_vessel6 = test_point[5]*np.ones((X_residual_vessel6.shape[0],1))
    X_test_vessel7 = test_point[6]*np.ones((X_residual_vessel7.shape[0],1))

    A_predicted_vessel1, U_predicted_vessel1, p_predicted_vessel1  = model.predict_vessel1(X_test_vessel1, T_residual)
    A_predicted_vessel2, U_predicted_vessel2, p_predicted_vessel2  = model.predict_vessel2(X_test_vessel2, T_residual)
    A_predicted_vessel3, U_predicted_vessel3, p_predicted_vessel3  = model.predict_vessel3(X_test_vessel3, T_residual)
    A_predicted_vessel4, U_predicted_vessel4, p_predicted_vessel4  = model.predict_vessel4(X_test_vessel4, T_residual)
    A_predicted_vessel5, U_predicted_vessel5, p_predicted_vessel5  = model.predict_vessel5(X_test_vessel5, T_residual)
    A_predicted_vessel6, U_predicted_vessel6, p_predicted_vessel6  = model.predict_vessel6(X_test_vessel6, T_residual)
    A_predicted_vessel7, U_predicted_vessel7, p_predicted_vessel7  = model.predict_vessel7(X_test_vessel7, T_residual)
   
    fig1 = plt.figure(1,figsize=(22, 12), dpi=111, facecolor='w', frameon = False)
    fig2 = plt.figure(2,figsize=(22, 12), dpi=110, facecolor='w', frameon = False)
    
    ax11 = fig1.add_subplot(241)  
    ax12 = fig1.add_subplot(242)  
    ax13 = fig1.add_subplot(243)  
    ax14 = fig1.add_subplot(244)  
    ax15 = fig1.add_subplot(245)  
    ax16 = fig1.add_subplot(246)  
    ax17 = fig1.add_subplot(247)  
    
     
    ax21 = fig2.add_subplot(241)  
    ax22 = fig2.add_subplot(242)  
    ax23 = fig2.add_subplot(243)  
    ax24 = fig2.add_subplot(244)  
    ax25 = fig2.add_subplot(245)  
    ax26 = fig2.add_subplot(246)  
    ax27 = fig2.add_subplot(247)  

    ax11.plot(T_residual, U_predicted_vessel1,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')
    ax11.plot(t, velocity_test_vessel1,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel1')

    ax12.plot(T_residual, U_predicted_vessel2,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel2')
    ax12.plot(t, velocity_test_vessel2,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel2')

    ax13.plot(T_residual, U_predicted_vessel3,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel3')
    ax13.plot(t, velocity_test_vessel3,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel3')

    ax14.plot(T_residual, U_predicted_vessel4,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel4')
    ax14.plot(t, velocity_test_vessel4,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel4')

    ax15.plot(T_residual, U_predicted_vessel5,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel5')
    ax15.plot(t, velocity_test_vessel5,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel5')

    ax16.plot(T_residual, U_predicted_vessel6,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel6')
    ax16.plot(t, velocity_test_vessel6,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel6')

    ax17.plot(T_residual, U_predicted_vessel7,'bo',linewidth=1, markersize=0.5, label='Predicted velocity Vessel7')
    ax17.plot(t, velocity_test_vessel7,'ro',linewidth=1, markersize=0.5, label='Reference velocity Vessel7')

    fig1.suptitle('Comparative velocity')
    ax15.set_xlabel("t in s")
    ax11.set_ylabel("Velocity in m/s")
    ax15.set_ylabel("Velocity in m/s")

    ax21.plot(T_residual, p_predicted_vessel1,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
    ax21.plot(t, pressure_test_vessel1,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel1')

    ax22.plot(T_residual, p_predicted_vessel2,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel2')
    ax22.plot(t, pressure_test_vessel2,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel2')

    ax23.plot(T_residual, p_predicted_vessel3,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel3')
    ax23.plot(t, pressure_test_vessel3,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel3')

    ax24.plot(T_residual, p_predicted_vessel4,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel4')
    ax24.plot(t, pressure_test_vessel4,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel4')

    ax25.plot(T_residual, p_predicted_vessel5,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel5')
    ax25.plot(t, pressure_test_vessel5,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel5')

    ax26.plot(T_residual, p_predicted_vessel6,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel6')
    ax26.plot(t, pressure_test_vessel6,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel6')

    ax27.plot(T_residual, p_predicted_vessel7,'bo',linewidth=1, markersize=0.5, label='Predicted pressure Vessel7')
    ax27.plot(t, pressure_test_vessel7,'ro',linewidth=1, markersize=0.5, label='Reference pressure Vessel7')

    fig2.suptitle('Comparative pressure')
    ax25.set_xlabel("t in s")
    ax21.set_ylabel("Pressure in Pa")
    ax25.set_ylabel("Pressure in Pa")
    
    ax21.set_ylim(6000,15000)   
    ax22.set_ylim(6000,15000)
    ax23.set_ylim(6000,15000)
    ax24.set_ylim(6000,15000)
    ax25.set_ylim(6000,15000)
    ax26.set_ylim(6000,15000)
    ax27.set_ylim(6000,15000)

    ax12.legend(loc='upper center', bbox_to_anchor=(1.18, 1.23),
          ncol=2, fancybox=True, shadow=True)
    
    ax22.legend(loc='upper center', bbox_to_anchor=(1.18, 1.23),
          ncol=2, fancybox=True, shadow=True)
    
    fig1.savefig("Comparative_Velocity.png")
    fig2.savefig("Comparative_Pressure.png")
    
