"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_2D, Gradient_Velocity_2D, \
                      tf_session, mean_squared_error, relative_error

class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _inlet: input-output data at the inlet
    # _cyl: points used to regress the no slip boundary condition
    # _star: preditions
    
    def __init__(self, t_data, x_data, y_data, c_data,
                       t_eqns, x_eqns, y_eqns,
                       t_inlet, x_inlet, y_inlet, u_inlet, v_inlet,
                       t_cyl, x_cyl, y_cyl,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        
        # data
        [self.t_data, self.x_data, self.y_data, self.c_data] = [t_data, x_data, y_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
        [self.t_inlet, self.x_inlet, self.y_inlet, self.u_inlet, self.v_inlet] = [t_inlet, x_inlet, y_inlet, u_inlet, v_inlet]
        [self.t_cyl, self.x_cyl, self.y_cyl] = [t_cyl, x_cyl, y_cyl]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.t_inlet_tf, self.x_inlet_tf, self.y_inlet_tf, self.u_inlet_tf, self.v_inlet_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_cyl_tf, self.x_cyl_tf, self.y_cyl_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        
        # physics "uninformed" neural networks
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_cuvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)
        
        # physics "uninformed" neural networks (data at the inlet)
        [_,
         self.u_inlet_pred,
         self.v_inlet_pred,
         _] = self.net_cuvp(self.t_inlet_tf,
                            self.x_inlet_tf,
                            self.y_inlet_tf)
        
        # physics "uninformed" neural networks (data on the cylinder)
        [_,
         self.u_cyl_pred,
         self.v_cyl_pred,
         _] = self.net_cuvp(self.t_cyl_tf,
                            self.x_cyl_tf,
                            self.y_cyl_tf)
        
        # physics "informed" neural networks
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.p_eqns_pred] = self.net_cuvp(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf)
        
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred] = Navier_Stokes_2D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.Pec,
                                               self.Rey)
        
        # gradients required for the lift and drag forces
        [self.u_x_eqns_pred,
         self.v_x_eqns_pred,
         self.u_y_eqns_pred,
         self.v_y_eqns_pred] = Gradient_Velocity_2D(self.u_eqns_pred,
                                                    self.v_eqns_pred,
                                                    self.x_eqns_tf,
                                                    self.y_eqns_tf)
        
        # loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                    mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                    mean_squared_error(self.u_cyl_pred, 0.0) + \
                    mean_squared_error(self.v_cyl_pred, 0.0) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0)
        
        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    def train(self, total_time, learning_rate):
        
        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             c_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.c_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:])


            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.t_inlet_tf: self.t_inlet,
                       self.x_inlet_tf: self.x_inlet,
                       self.y_inlet_tf: self.y_inlet,
                       self.u_inlet_tf: self.u_inlet,
                       self.v_inlet_tf: self.v_inlet,
                       self.t_cyl_tf: self.t_cyl,
                       self.x_cyl_tf: self.x_cyl,
                       self.y_cyl_tf: self.y_cyl,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    def predict(self, t_star, x_star, y_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, p_star
    
    def predict_drag_lift(self, t_cyl):
        
        viscosity = (1.0/self.Rey)
        
        theta = np.linspace(0.0,2*np.pi,200)[:,None] # N x 1
        d_theta = theta[1,0] - theta[0,0]
        x_cyl = 0.5*np.cos(theta) # N x 1
        y_cyl = 0.5*np.sin(theta) # N x 1
            
        N = x_cyl.shape[0]
        T = t_cyl.shape[0]
        
        T_star = np.tile(t_cyl, (1,N)).T # N x T
        X_star = np.tile(x_cyl, (1,T)) # N x T
        Y_star = np.tile(y_cyl, (1,T)) # N x T
        
        t_star = np.reshape(T_star,[-1,1]) # NT x 1
        x_star = np.reshape(X_star,[-1,1]) # NT x 1
        y_star = np.reshape(Y_star,[-1,1]) # NT x 1
        
        tf_dict = {self.t_eqns_tf: t_star, self.x_eqns_tf: x_star, self.y_eqns_tf: y_star}
        
        [p_star,
         u_x_star,
         u_y_star,
         v_x_star,
         v_y_star] = self.sess.run([self.p_eqns_pred,
                                    self.u_x_eqns_pred,
                                    self.u_y_eqns_pred,
                                    self.v_x_eqns_pred,
                                    self.v_y_eqns_pred], tf_dict)
        
        P_star = np.reshape(p_star, [N,T]) # N x T
        P_star = P_star - np.mean(P_star, axis=0)
        U_x_star = np.reshape(u_x_star, [N,T]) # N x T
        U_y_star = np.reshape(u_y_star, [N,T]) # N x T
        V_x_star = np.reshape(v_x_star, [N,T]) # N x T
        V_y_star = np.reshape(v_y_star, [N,T]) # N x T
    
        INT0 = (-P_star[0:-1,:] + 2*viscosity*U_x_star[0:-1,:])*X_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*Y_star[0:-1,:]
        INT1 = (-P_star[1: , :] + 2*viscosity*U_x_star[1: , :])*X_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*Y_star[1: , :]
            
        F_D = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
    
        
        INT0 = (-P_star[0:-1,:] + 2*viscosity*V_y_star[0:-1,:])*Y_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*X_star[0:-1,:]
        INT1 = (-P_star[1: , :] + 2*viscosity*V_y_star[1: , :])*Y_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*X_star[1: , :]
            
        F_L = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
            
        return F_D, F_L
    
if __name__ == "__main__":
    
    batch_size = 10000
    
    layers = [3] + 10*[4*50] + [4]
    
    # Load Data
    data = scipy.io.loadmat('../Data/Cylinder2D_Re200Pec2000_Dirichlet_Streaks.mat')
    
    t_star = data['t_data'] # T x 1
    x_star = data['x_data'] # N x 1
    y_star = data['y_data'] # N x 1
    
    T = t_star.shape[0]
    N = x_star.shape[0]
    
    U_star = data['U_data'] # N x T
    V_star = data['V_data'] # N x T
    P_star = data['P_data'] # N x T
    C_star = data['C_data'] # N x T
    
    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    
    t = T_star.flatten()[:,None] # NT x 1
    x = X_star.flatten()[:,None] # NT x 1
    y = Y_star.flatten()[:,None] # NT x 1
    u = U_star.flatten()[:,None] # NT x 1
    v = V_star.flatten()[:,None] # NT x 1
    p = P_star.flatten()[:,None] # NT x 1
    c = C_star.flatten()[:,None] # NT x 1
    
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
    
    T_data = T # int(sys.argv[1])
    N_data = N # int(sys.argv[2])
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
        
    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    # Training Data on velocity (inlet)
    t_inlet = t[x == x.min()][:,None]
    x_inlet = x[x == x.min()][:,None]
    y_inlet = y[x == x.min()][:,None]
    u_inlet = u[x == x.min()][:,None]
    v_inlet = v[x == x.min()][:,None]
    
    # Training Data on velocity (cylinder)
    t_cyl = t[x**2 + y**2 == 0.25][:,None]
    x_cyl = x[x**2 + y**2 == 0.25][:,None]
    y_cyl = y[x**2 + y**2 == 0.25][:,None]
    
    # Training
    model = HFM(t_data, x_data, y_data, c_data,
                t_eqns, x_eqns, y_eqns,
                t_inlet, x_inlet, y_inlet, u_inlet, v_inlet,
                t_cyl, x_cyl, y_cyl,
                layers, batch_size,
                Pec = 2000, Rey = 200)
    
    model.train(total_time = 40, learning_rate=1e-3)

    F_D, F_L = model.predict_drag_lift(t_star)
    
    # Test Data
    snap = np.array([100])
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    
    c_test = C_star[:,snap]
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    p_test = P_star[:,snap]
    
    # Prediction
    c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

    print('Error c: %e' % (error_c))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
    
    ################# Save Data ###########################
    
    C_pred = 0*C_star
    U_pred = 0*U_star
    V_pred = 0*V_star
    P_pred = 0*P_star
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        
        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
    
        # Prediction
        c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
        
        C_pred[:,snap:snap+1] = c_pred
        U_pred[:,snap:snap+1] = u_pred
        V_pred[:,snap:snap+1] = v_pred
        P_pred[:,snap:snap+1] = p_pred
    
        # Error
        error_c = relative_error(c_pred, c_test)
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))
    
        print('Error c: %e' % (error_c))
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))
    
    scipy.io.savemat('../Results/Cylinder2D_No_Slip_Dirichlet_Streaks_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'P_pred':P_pred, 'F_L':F_L, 'F_D':F_D})
