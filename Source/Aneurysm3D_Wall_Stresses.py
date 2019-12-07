"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_3D, Shear_Stress_3D, \
                      tf_session, mean_squared_error, relative_error
    
class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions
    
    def __init__(self, t_data, x_data, y_data, z_data, c_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        
        # data
        [self.t_data, self.x_data, self.y_data, self.z_data, self.c_data] = [t_data, x_data, y_data, z_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.nx_eqns_tf, self.ny_eqns_tf, self.nz_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        
        # physics "uninformed" neural networks
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                            self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)
         
        # physics "informed" neural networks
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.w_eqns_pred,
         self.p_eqns_pred] = self.net_cuvwp(self.t_eqns_tf,
                                            self.x_eqns_tf,
                                            self.y_eqns_tf,
                                            self.z_eqns_tf)
        
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred,
         self.e5_eqns_pred] = Navier_Stokes_3D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.w_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.z_eqns_tf,
                                               self.Pec,
                                               self.Rey)

        [self.sx_eqns_pred,
         self.sy_eqns_pred,
         self.sz_eqns_pred] = Shear_Stress_3D(self.u_eqns_pred,
                                              self.v_eqns_pred,
                                              self.w_eqns_pred,
                                              self.x_eqns_tf,
                                              self.y_eqns_tf,
                                              self.z_eqns_tf,
                                              self.nx_eqns_tf,
                                              self.ny_eqns_tf,
                                              self.nz_eqns_tf,
                                              self.Rey)
        
        # loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0) + \
                    mean_squared_error(self.e5_eqns_pred, 0.0)
        
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
            
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             z_data_batch,
             c_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.z_data[idx_data,:],
                              self.c_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch,
             z_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:],
                              self.z_eqns[idx_eqns,:])


            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
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
    
    def predict(self, t_star, x_star, y_star, z_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, w_star, p_star
    
    def predict_shear(self, t_star, x_star, y_star, z_star, nx_star, ny_star, nz_star):
        
        tf_dict = {self.t_eqns_tf: t_star, self.x_eqns_tf: x_star, self.y_eqns_tf: y_star, self.z_eqns_tf: z_star,
                   self.nx_eqns_tf: nx_star, self.ny_eqns_tf: ny_star, self.nz_eqns_tf: nz_star}
        
        sx_star = self.sess.run(self.sx_eqns_pred, tf_dict)
        sy_star = self.sess.run(self.sy_eqns_pred, tf_dict)
        sz_star = self.sess.run(self.sz_eqns_pred, tf_dict)
        
        return sx_star, sy_star, sz_star
    
if __name__ == "__main__":
    
    batch_size = 10000
    
    layers = [4] + 10*[5*50] + [5]
    
    # Load Shear Data
    data_shear = scipy.io.loadmat('../Data/real_aneurysm_shear.mat')
    
    xb_star = data_shear['xb_star'] 
    yb_star = data_shear['yb_star']
    zb_star = data_shear['zb_star']
    nx_star = data_shear['nx_star']
    ny_star = data_shear['ny_star']
    nz_star = data_shear['nz_star']
    Sx_star = data_shear['Sx_star']
    Sy_star = data_shear['Sy_star']
    Sz_star = data_shear['Sz_star']
    
    # Load Data
    data = scipy.io.loadmat('../Data/real_aneurysm.mat')
    
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1
    z_star = data['z_star'] # N x 1
    
    T = t_star.shape[0]
    N = x_star.shape[0]
        
    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    W_star = data['W_star'] # N x T
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T
    
    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    Z_star = np.tile(z_star, (1,T)) # N x T
    
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
    
    T_data = T
    N_data = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    # Training
    model = HFM(t_data, x_data, y_data, z_data, c_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                layers, batch_size,
                Pec = 1.0/0.01, Rey = 1.0/0.01)
    
    model.train(total_time = 40, learning_rate=1e-3)
    
    # Test Data
    snap = np.array([150])
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    z_test = Z_star[:,snap]
    
    c_test = C_star[:,snap]
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    w_test = W_star[:,snap]
    p_test = P_star[:,snap]
    
    sx_test = Sx_star[:,snap]
    sy_test = Sy_star[:,snap]
    sz_test = Sz_star[:,snap]
    
    # Prediction
    c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
    
    # Shear
    sx_pred, sy_pred, sz_pred = model.predict_shear(t_test[0] + 0.0*xb_star,
                                                    xb_star, yb_star, zb_star,
                                                    nx_star, ny_star, nz_star)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_w = relative_error(w_pred, w_test)
    error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

    print('Error c: %e' % (error_c))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error w: %e' % (error_w))
    print('Error p: %e' % (error_p))
    sys.stdout.flush()
    
    # Error
    error_sx = relative_error(sx_pred, sx_test)
    error_sy = relative_error(sy_pred, sy_test)
    error_sz = relative_error(sz_pred, sz_test)
    
    print('Error sx: %e' % (error_sx))
    print('Error sy: %e' % (error_sy))
    print('Error sz: %e' % (error_sz))
    sys.stdout.flush()
    
    ################# Save Data ###########################
    
    C_pred = 0*C_star
    U_pred = 0*U_star
    V_pred = 0*V_star
    W_pred = 0*W_star
    P_pred = 0*P_star
    Sx_pred = 0*Sx_star
    Sy_pred = 0*Sy_star
    Sz_pred = 0*Sz_star
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
        
        sx_test = Sx_star[:,snap:snap+1]
        sy_test = Sy_star[:,snap:snap+1]
        sz_test = Sz_star[:,snap:snap+1]
    
        # Prediction
        c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
        # Shear
        sx_pred, sy_pred, sz_pred = model.predict_shear(t_test[0] + 0.0*xb_star,
                                                        xb_star, yb_star, zb_star,
                                                        nx_star, ny_star, nz_star)
        
        C_pred[:,snap:snap+1] = c_pred
        U_pred[:,snap:snap+1] = u_pred
        V_pred[:,snap:snap+1] = v_pred
        W_pred[:,snap:snap+1] = w_pred
        P_pred[:,snap:snap+1] = p_pred
        
        Sx_pred[:,snap:snap+1] = sx_pred
        Sy_pred[:,snap:snap+1] = sy_pred
        Sz_pred[:,snap:snap+1] = sz_pred
        
        # Error
        error_c = relative_error(c_pred, c_test)
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_w = relative_error(w_pred, w_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))
    
        print('Error c: %e' % (error_c))
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('Error p: %e' % (error_p))
        sys.stdout.flush()
    
        # Error
        error_sx = relative_error(sx_pred, sx_test)
        error_sy = relative_error(sy_pred, sy_test)
        error_sz = relative_error(sz_pred, sz_test)
        
        print('Error sx: %e' % (error_sx))
        print('Error sy: %e' % (error_sy))
        print('Error sz: %e' % (error_sz))
        sys.stdout.flush()
    
    scipy.io.savemat('../Results/Aneurysm3D_Wall_Stresses_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'W_pred':W_pred, 'P_pred':P_pred,
                      'Sx_pred':Sx_pred, 'Sy_pred':Sy_pred, 'Sz_pred':Sz_pred})
