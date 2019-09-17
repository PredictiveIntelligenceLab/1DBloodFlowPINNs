import tensorflow as tf
import numpy as np
import timeit

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_measurement_aorta1, X_measurement_carotid,\
                       X_measurement_aorta3, X_measurement_aorta4,
                       T_measurement, T_initial, 
                       A_training_aorta1,  U_training_aorta1,
                          A_training_carotid, U_training_carotid,
                          A_training_aorta3,  U_training_aorta3,
                          A_training_aorta4,  U_training_aorta4, 
                          X_residual_aorta1, 
                          X_residual_carotid, 
                          X_residual_aorta3, 
                          X_residual_aorta4,
                          T_residual,layers,bif_points):

        self.A_01 = 2.293820e-04
        self.A_02 = 2.623127e-05
        self.A_03 = 2.411245e-04
        
        self.rho = 1060.                   
        self.U = 1e+1

        self.L = np.sqrt(0.333*(self.A_01 + self.A_02 + self.A_03))
        self.T = self.L/self.U
        self.p0 = self.rho*self.U**2        

        self.A0 = self.L**2     
        
        X_measurement_aorta1 = X_measurement_aorta1/self.L
        X_measurement_carotid = X_measurement_carotid/self.L
        X_measurement_aorta3 = X_measurement_aorta3/self.L
        X_measurement_aorta4 = X_measurement_aorta4/self.L
        
        X_residual_aorta1 = X_residual_aorta1/self.L
        X_residual_carotid = X_residual_carotid/self.L
        X_residual_aorta3 = X_residual_aorta3/self.L
        X_residual_aorta4 = X_residual_aorta4/self.L
        
        T_measurement  = T_measurement/self.T
        T_residual  = T_residual/self.T
        T_initial  = T_initial/self.T
        
        # Normalize inputs
        self.Xmean1, self.Xstd1 = X_residual_aorta1.mean(0), X_residual_aorta1.std(0)
        self.Xmean2, self.Xstd2 = X_residual_carotid.mean(0), X_residual_carotid.std(0)
        self.Xmean3, self.Xstd3 = X_residual_aorta3.mean(0), X_residual_aorta3.std(0)
        self.Xmean4, self.Xstd4 = X_residual_aorta4.mean(0), X_residual_aorta4.std(0)

        self.Tmean, self.Tstd = T_residual.mean(0), T_residual.std(0)
        
        self.jac_x1 = 1.0/self.Xstd1
        self.jac_x2 = 1.0/self.Xstd2
        self.jac_x3 = 1.0/self.Xstd3
        self.jac_x4 = 1.0/self.Xstd4

        self.jac_t = 1.0/self.Tstd
        

        self.X_f1 = (X_residual_aorta1 - self.Xmean1)/self.Xstd1
        self.X_u1 = (X_measurement_aorta1 - self.Xmean1)/self.Xstd1
        
        self.X_f2 = (X_residual_carotid - self.Xmean2)/self.Xstd2
        self.X_u2 = (X_measurement_carotid - self.Xmean2)/self.Xstd2
        
        self.X_f3 = (X_residual_aorta3 - self.Xmean3)/self.Xstd3
        self.X_u3 = (X_measurement_aorta3 - self.Xmean3)/self.Xstd3

        self.X_f4 = (X_residual_aorta4 - self.Xmean4)/self.Xstd4
        self.X_u4 = (X_measurement_aorta4 - self.Xmean4)/self.Xstd4

        self.T_u = (T_measurement - self.Tmean)/self.Tstd
        self.T_f = (T_residual - self.Tmean)/self.Tstd
        self.T_i = (T_initial - self.Tmean)/self.Tstd
        
        self.layers = layers
        
        self.A_u1 = A_training_aorta1 
        self.u_u1 = U_training_aorta1
        
        self.A_u2 = A_training_carotid
        self.u_u2 = U_training_carotid

        self.A_u3 = A_training_aorta3 
        self.u_u3 = U_training_aorta3

        self.A_u4 = A_training_aorta4 
        self.u_u4 = U_training_aorta4
        
        X1_fm = bif_points[0]/self.L
        X2_fm = bif_points[0]/self.L
        X3_fm1 = bif_points[0]/self.L
        X3_fm2 = bif_points[1]/self.L
        
        
        bif_p1 = (X1_fm - self.Xmean1)/self.Xstd1
        bif_p2 = (X2_fm - self.Xmean2)/self.Xstd2
        bif_p31 = (X3_fm1 - self.Xmean3)/self.Xstd3
        bif_p32 = (X3_fm2 - self.Xmean3)/self.Xstd3
        bif_p4 = (X3_fm2 - self.Xmean4)/self.Xstd4        
       
        X1max = bif_p1[0]
        X2min = bif_p2[0]
        X3min = bif_p31[0]
        X3max = bif_p32[0]
        X4min = bif_p4[0]

        # Initialize network weights and biases        
        self.weights1, self.biases1 = self.initialize_NN(layers)
        self.weights2, self.biases2 = self.initialize_NN(layers)
        self.weights3, self.biases3 = self.initialize_NN(layers)
        self.weights4, self.biases4 = self.initialize_NN(layers)
                       
        # Define placeholders and computational graph
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        self.X1_fm = tf.constant([X1max], shape = [1024,1], dtype=tf.float32)
        self.X2_fm = tf.constant([X2min], shape = [1024,1], dtype=tf.float32)
        self.X3_fml = tf.constant([X3min], shape = [1024,1], dtype=tf.float32)
        self.X3_fmu = tf.constant([X3max], shape = [1024,1], dtype=tf.float32)
        self.X4_fm = tf.constant([X4min], shape = [1024,1], dtype=tf.float32)
        
        self.A_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.A_u1.shape[1]))
        self.u_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.u_u1.shape[1]))
        
        self.A_u_tf2 = tf.placeholder(tf.float32, shape=(None, self.A_u2.shape[1]))
        self.u_u_tf2 = tf.placeholder(tf.float32, shape=(None, self.u_u2.shape[1]))

        self.A_u_tf3 = tf.placeholder(tf.float32, shape=(None, self.A_u3.shape[1]))
        self.u_u_tf3 = tf.placeholder(tf.float32, shape=(None, self.u_u3.shape[1]))

        self.A_u_tf4 = tf.placeholder(tf.float32, shape=(None, self.A_u4.shape[1]))
        self.u_u_tf4 = tf.placeholder(tf.float32, shape=(None, self.u_u4.shape[1]))
                
        self.X_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.X_u1.shape[1]))
        self.X_u_tf2 = tf.placeholder(tf.float32, shape=(None, self.X_u2.shape[1]))
        self.X_u_tf3 = tf.placeholder(tf.float32, shape=(None, self.X_u3.shape[1]))
        self.X_u_tf4 = tf.placeholder(tf.float32, shape=(None, self.X_u4.shape[1]))
        
        self.t_u_tf = tf.placeholder(tf.float32,  shape=(None, self.T_u.shape[1]))
        self.t_i_tf = tf.placeholder(tf.float32,  shape=(None, self.T_i.shape[1]))

        self.X_f_tf1 = tf.placeholder(tf.float32, shape=(None, self.X_f1.shape[1]))
        self.X_f_tf2 = tf.placeholder(tf.float32, shape=(None, self.X_f2.shape[1]))
        self.X_f_tf3 = tf.placeholder(tf.float32, shape=(None, self.X_f3.shape[1]))
        self.X_f_tf4 = tf.placeholder(tf.float32, shape=(None, self.X_f4.shape[1]))

        self.t_f_tf = tf.placeholder(tf.float32, shape=(None, self.T_f.shape[1]))
        
        self.A_u_pred1, self.u_u_pred1, _  = self.neural_net_aorta1(self.X_u_tf1, self.t_u_tf)
        self.A_u_pred2, self.u_u_pred2, _  = self.neural_net_carotid(self.X_u_tf2, self.t_u_tf)
        self.A_u_pred3, self.u_u_pred3, _  = self.neural_net_aorta3(self.X_u_tf3, self.t_i_tf)
        self.A_u_pred4, self.u_u_pred4, _  = self.neural_net_aorta4(self.X_u_tf4, self.t_u_tf)
        
        self.A_f_pred1, self.u_f_pred1, self.p_f_pred1  = self.neural_net_aorta1(self.X_f_tf1, self.t_f_tf)
        self.A_f_pred2, self.u_f_pred2, self.p_f_pred2  = self.neural_net_carotid(self.X_f_tf2, self.t_f_tf)
        self.A_f_pred3, self.u_f_pred3, self.p_f_pred3  = self.neural_net_aorta3(self.X_f_tf3, self.t_f_tf)
        self.A_f_pred4, self.u_f_pred4, self.p_f_pred4  = self.neural_net_aorta4(self.X_f_tf4, self.t_f_tf)
        
        self.r_A1, self.r_u1, self.r_p1  = self.pinn_aorta1(self.X_f_tf1, self.t_f_tf)
        self.r_A2, self.r_u2, self.r_p2  = self.pinn_carotid(self.X_f_tf2, self.t_f_tf)
        self.r_A3, self.r_u3, self.r_p3  = self.pinn_aorta3(self.X_f_tf3, self.t_f_tf)
        self.r_A4, self.r_u4, self.r_p4  = self.pinn_aorta4(self.X_f_tf4, self.t_f_tf)
               
        self.loss_A1, self.loss_u1                 = self.compute_measurement_loss_aorta1(self.A_u_pred1, self.u_u_pred1)
        self.loss_rA1, self.loss_ru1, self.loss_rp1 = self.compute_residual_loss_aorta1 (self.r_A1, self.r_u1, self.r_p1)
        
        self.loss_A2, self.loss_u2                 = self.compute_measurement_loss_carotid(self.A_u_pred2, self.u_u_pred2)
        self.loss_rA2, self.loss_ru2, self.loss_rp2 = self.compute_residual_loss_carotid (self.r_A2, self.r_u2, self.r_p2)

        self.loss_A3, self.loss_u3                 = self.compute_measurement_loss_aorta3(self.A_u_pred3, self.u_u_pred3)
        self.loss_rA3, self.loss_ru3, self.loss_rp3 = self.compute_residual_loss_aorta3 (self.r_A3, self.r_u3, self.r_p3)

        self.loss_A4, self.loss_u4                 = self.compute_measurement_loss_aorta4(self.A_u_pred4, self.u_u_pred4)
        self.loss_rA4, self.loss_ru4, self.loss_rp4 = self.compute_residual_loss_aorta4 (self.r_A4, self.r_u4, self.r_p4)
     
        self.loss_interface  = self.compute_interface_loss()
        
        self.loss_A = self.loss_A1 + self.loss_A2 + self.loss_A3 + self.loss_A4
        self.loss_u = self.loss_u1 + self.loss_u2 + self.loss_u3 + self.loss_u4
        
        self.loss_measurements = self.loss_A + self.loss_u
        
        self.loss_ru = self.loss_ru1 + self.loss_ru2 + self.loss_ru3 + self.loss_ru4
        self.loss_rA = self.loss_rA1 + self.loss_rA2 + self.loss_rA3 + self.loss_rA4
        self.loss_rp = self.loss_rp1 + self.loss_rp2 + self.loss_rp3 + self.loss_rp4
        self.loss_residual = self.loss_rA + self.loss_ru + self.loss_rp
        
        self.loss = self.loss_interface + self.loss_residual  + self.loss_measurements
        
        # Define optimizer        
        self.optimizer  = tf.train.AdamOptimizer(self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss)
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = False
        # Define Tensorflow session
        self.sess = tf.Session(config=config)
        
        # Initialize Tensorflow variables
        self.merged = tf.summary.merge_all()
        
        self.summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
   
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
       
           
    def neural_net(self, H, weights, biases, layers):
        num_layers = len(layers)  
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_aorta1(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights1,self.biases1,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p
    
    def neural_net_carotid(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights2,self.biases2,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p
    
    def neural_net_aorta3(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights3,self.biases3,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p

    def neural_net_aorta4(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights4,self.biases4,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p
   
    def compute_interface_loss(self):
        
         A1, u1, p1 = self.neural_net_aorta1(self.X1_fm,self.t_f_tf) # A*, u*, p*
         
         A2, u2, p2 = self.neural_net_carotid(self.X2_fm,self.t_f_tf) # A*, u*, p*
         
         A3, u3, p3 = self.neural_net_aorta3(self.X3_fml,self.t_f_tf) # A*, u*, p*
         
         A3u, u3u, p3u = self.neural_net_aorta3(self.X3_fmu,self.t_f_tf) # A*, u*, p*

         A4, u4, p4 = self.neural_net_aorta4(self.X4_fm,self.t_f_tf) # A*, u*, p*
         
         Q1 = A1*u1
         Q2 = A2*u2
         Q3 = A3*u3
         
         loss_mass = tf.reduce_mean(tf.square((Q1 - Q2 - Q3))) 
         
         p_1 = p1 + (0.5*u1**2)
         p_2 = p2 + (0.5*u2**2)
         p_3 = p3 + (0.5*u3**2)
         
         loss_press = tf.reduce_mean(tf.square( p_1 - p_2)) + tf.reduce_mean(tf.square( p_1 - p_3))
                                
                         
         loss_C = tf.reduce_mean(tf.square((u3u - u4))) + \
                             tf.reduce_mean(tf.square((A3u - A4))) + tf.reduce_mean(tf.square( p3u - p4))
                             
         return  loss_mass + loss_press + loss_C
     

    def get_equilibrium_cross_sectional_area_aorta_1(self, x):
        x = self.L*(self.Xstd1*x + self.Xmean1)
        X1 = 0.
        X2 = 0.04964
        denom = X2-X1
        x1 = 2.293820e-04
        x2 = 2.636589e-04
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y

    def get_equilibrium_cross_sectional_area_carotid(self, x):
        x = self.L*(self.Xstd2*x + self.Xmean2)
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

    def get_equilibrium_cross_sectional_area_aorta_3(self, x):
        x = self.L*(self.Xstd3*x + self.Xmean3)
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

    def get_equilibrium_cross_sectional_area_aorta_4(self, x):
        x = self.L*(self.Xstd4*x + self.Xmean4)
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

    def get_beta_aorta_1(self, x):
        x = self.L*(self.Xstd1*x + self.Xmean1)
        X1 = 0.
        X2 = 0.04964
        denom = X2-X1
        x1 = 2.472667e+06
        x2 = 2.151208e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def get_beta_carotid(self, x):
        x = self.L*(self.Xstd2*x + self.Xmean2)
        X1 = 0.04964
        X2 = 0.10284
        denom = X2-X1
        x1 =2.151208e+06
        x2 = 9.459836e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def get_beta_aorta_3(self, x):
        x = self.L*(self.Xstd3*x + self.Xmean3)
        X1 = 0.04964
        X2 = 0.1383
        denom = X2-X1
        x1 = 2.151208e+06
        x2 = 2.800526e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def get_beta_aorta_4(self, x):
        x = self.L*(self.Xstd4*x + self.Xmean4)
        X1 = 0.1383
        X2 = 0.17056
        denom = X2-X1
        x1 = 2.800526e+06
        x2 = 2.528670e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
     
    def pinn_aorta1(self, x, t):
        
        A, u, p = self.neural_net_aorta1(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_01 = self.get_equilibrium_cross_sectional_area_aorta_1(x)
        beta1 = self.get_beta_aorta_1(x)
        
        r_p  = 10000. + beta1*(tf.sqrt(A*self.A0) - tf.sqrt(A_01)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x1

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x1
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x1
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p
    
    def pinn_carotid(self, x, t):
        
        A, u, p = self.neural_net_carotid(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_02 = self.get_equilibrium_cross_sectional_area_carotid(x)
        beta2 = self.get_beta_carotid(x)
        
        r_p  = 8.5e+3 + beta2*(tf.sqrt(A*self.A0) - tf.sqrt(A_02)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x2

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x2
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x2
                
        r_A = A_t + u*A_x +  A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p
    
    def pinn_aorta3(self, x, t):
        
        A, u, p = self.neural_net_aorta3(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_03 = self.get_equilibrium_cross_sectional_area_aorta_3(x)
        beta3 = self.get_beta_aorta_3(x)

        r_p  = 10000. + beta3*(tf.sqrt(A*self.A0) - tf.sqrt(A_03)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x3

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x3
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x3
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p

    def pinn_aorta4(self, x, t):
        
        A, u, p = self.neural_net_aorta4(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_04 = self.get_equilibrium_cross_sectional_area_aorta_4(x)
        beta4 = self.get_beta_aorta_4(x)
        
        r_p  = 10000. + beta4*(tf.sqrt(A*self.A0) - tf.sqrt(A_04)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x4

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x4
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x4
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p

    def compute_residual_loss_aorta1(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred1 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp

    def compute_residual_loss_carotid(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred2 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp

    def compute_residual_loss_aorta3(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred3 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp 

    def compute_residual_loss_aorta4(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred4 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp 

    def compute_measurement_loss_aorta1(self, A_u, u_u):
    
        loss_A = tf.reduce_mean(tf.square((self.A_u1 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u1 - u_u*self.U)/self.U))

        return loss_A, loss_u

    def compute_measurement_loss_carotid(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u2 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u2 - u_u*self.U)/self.U))

        return loss_A, loss_u

    def compute_measurement_loss_aorta3(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u3 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u3 - u_u*self.U)/self.U))

        return loss_A, loss_u

    def compute_measurement_loss_aorta4(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u4 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u4 - u_u*self.U)/self.U))

        return loss_A, loss_u
      
    
    def fetch_minibatch(self, X1_f, X2_f, X3_f ,X4_f, t_f, N_f_batch):        
        N_f = X1_f.shape[0]
        idx_f = np.random.choice(N_f, N_f_batch, replace=False)
        X1_f_batch = X1_f[idx_f,:]
        X2_f_batch = X2_f[idx_f,:]
        X3_f_batch = X3_f[idx_f,:]
        X4_f_batch = X4_f[idx_f,:]

        t_f_batch = t_f[idx_f,:]        
        return  X1_f_batch, X2_f_batch, X3_f_batch, X4_f_batch, t_f_batch
             
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 20000, learning_rate = 1e-3): 
        
    
        start_time = timeit.default_timer()
        
        for it in range(nIter):
            
            X1_f_batch, X2_f_batch, X3_f_batch, X4_f_batch, T_f_batch = \
                    self.fetch_minibatch(self.X_f1, self.X_f2, self.X_f3, self.X_f4, self.T_f,\
                                         N_f_batch = 1024)
            self.T_f_b = T_f_batch
        # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_u_tf1: self.X_u1,  
                       self.X_u_tf2: self.X_u2, 
                       self.X_u_tf3: self.X_u3, 
                       self.X_u_tf4: self.X_u4, 
                       self.X_f_tf1: X1_f_batch,
                       self.X_f_tf2: X2_f_batch, 
                       self.X_f_tf3: X3_f_batch,
                       self.X_f_tf4: X4_f_batch,
                       self.t_f_tf:  T_f_batch, 
                       self.t_u_tf:  self.T_u,
                       self.t_i_tf:  self.T_i,
                       self.A_u_tf1: self.A_u1, self.u_u_tf1: self.u_u1, 
                       self.A_u_tf2: self.A_u2, self.u_u_tf2: self.u_u2,
                       self.A_u_tf3: self.A_u3, self.u_u_tf3: self.u_u3,
                       self.A_u_tf4: self.A_u4, self.u_u_tf4: self.u_u4,
                       self.learning_rate: learning_rate}

                 
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value, loss_A, loss_u, loss_r, loss_rp, loss_c  = self.sess.run([self.loss, self.loss_A, \
                                                self.loss_u, self.loss_ru+self.loss_rA, self.loss_rp, self.loss_interface], tf_dict)
                print('It: %d, Loss: %.3e, Loss_A: %.3e, Loss_u: %.3e, Loss_r: %.3e, Loss_p: %.3e\
                                           Loss_c: %.3e, Time: %.2f' % 
                      (it, loss_value, loss_A, loss_u, loss_r, loss_rp, loss_c, elapsed))
                start_time = timeit.default_timer()
                                
    # Evaluates predictions at test points           
    def predict_aorta1(self, X1, t): 
        X1 = X1/self.L
        t  = t/self.T
        X1 = (X1 - self.Xmean1)/self.Xstd1

        t = (t - self.Tmean)/self.Tstd
        tf_dict1 = {self.X_f_tf1: X1, self.t_f_tf: t}    
       
        A_star1 = self.sess.run(self.A_f_pred1, tf_dict1) 
        u_star1 = self.sess.run(self.u_f_pred1, tf_dict1) 
        p_star1 = self.sess.run(self.p_f_pred1, tf_dict1) 
                
        A_star1 = A_star1*self.A0
        u_star1 = u_star1*self.U
        p_star1 = p_star1*self.p0
              
        return A_star1, u_star1, p_star1

    def predict_carotid(self, X2, t):     
        X2 = X2/self.L
        t  = t/self.T

        X2 = (X2 - self.Xmean2)/self.Xstd2

        t = (t - self.Tmean)/self.Tstd        
        tf_dict2 = {self.X_f_tf2: X2, self.t_f_tf: t}    
       
        A_star2 = self.sess.run(self.A_f_pred2, tf_dict2) 
        u_star2 = self.sess.run(self.u_f_pred2, tf_dict2) 
        p_star2 = self.sess.run(self.p_f_pred2, tf_dict2) 
                
        A_star2 = A_star2*self.A0
        u_star2 = u_star2*self.U
        p_star2 = p_star2*self.p0
              
        return A_star2, u_star2, p_star2
    
    def predict_aorta3(self, X3, t):     
        X3 = X3/self.L
        t  = t/self.T

        X3 = (X3 - self.Xmean3)/self.Xstd3
        t = (t - self.Tmean)/self.Tstd
        
        tf_dict3 = {self.X_f_tf3: X3, self.t_f_tf: t}    
       
        A_star3 = self.sess.run(self.A_f_pred3, tf_dict3) 
        u_star3 = self.sess.run(self.u_f_pred3, tf_dict3) 
        p_star3 = self.sess.run(self.p_f_pred3, tf_dict3) 
                
        A_star3 = A_star3*self.A0
        u_star3 = u_star3*self.U
        p_star3 = p_star3*self.p0
              
        return A_star3, u_star3, p_star3
    
    def predict_aorta4(self, X4, t):     
        X4 = X4/self.L
        t  = t/self.T

        X4 = (X4 - self.Xmean4)/self.Xstd4
        t = (t - self.Tmean)/self.Tstd
        
        tf_dict4 = {self.X_f_tf4: X4, self.t_f_tf: t}    
       
        A_star4 = self.sess.run(self.A_f_pred4, tf_dict4) 
        u_star4 = self.sess.run(self.u_f_pred4, tf_dict4) 
        p_star4 = self.sess.run(self.p_f_pred4, tf_dict4) 
                
        A_star4 = A_star4*self.A0
        u_star4 = u_star4*self.U
        p_star4 = p_star4*self.p0
              
        return A_star4, u_star4, p_star4

