"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_2D, Strain_Rate_2D, \
                      tf_session, mean_squared_error, relative_error

class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions

    def __init__(self, t_data, x_data, y_data, c_data,
                       t_eqns, x_eqns, y_eqns,
                       layers, batch_size):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = tf.Variable(15.0, dtype=tf.float32, trainable = True)
        self.Rey = tf.Variable(5.0, dtype=tf.float32, trainable = True)
                
        # data
        [self.t_data, self.x_data, self.y_data, self.c_data] = [t_data, x_data, y_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        
        # physics "uninformed" neural networks
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_cuvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)
         
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
        
        [self.eps11dot_eqns_pred,
         self.eps12dot_eqns_pred,
         self.eps22dot_eqns_pred] = Strain_Rate_2D(self.u_eqns_pred,
                                                   self.v_eqns_pred,
                                                   self.x_eqns_tf,
                                                   self.y_eqns_tf)
        
        # loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
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
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 Pec_value,
                 Rey_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.Pec,
                                                       self.Rey,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Pec: %.3f, Rey: %.3f, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, Pec_value, Rey_value, elapsed, running_time, learning_rate_value))
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
    
    def predict_eps_dot(self, t_star, x_star, y_star):
                
        tf_dict = {self.t_eqns_tf: t_star, self.x_eqns_tf: x_star, self.y_eqns_tf: y_star}
        
        eps11dot_star = self.sess.run(self.eps11dot_eqns_pred, tf_dict)
        eps12dot_star = self.sess.run(self.eps12dot_eqns_pred, tf_dict)
        eps22dot_star = self.sess.run(self.eps22dot_eqns_pred, tf_dict)
        
        return eps11dot_star, eps12dot_star, eps22dot_star
        
if __name__ == "__main__": 
      
    batch_size = 10000
     
    layers = [3] + 10*[4*50] + [4]
    
    # Load Data
    data = scipy.io.loadmat('../Data/Stenosis2D.mat')
          
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]

    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T    
            
    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T    
    
    ######################################################################
    ######################## Noiseles Data ###############################
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
        
    # Training
    model = HFM(t_data, x_data, y_data, c_data,
                t_eqns, x_eqns, y_eqns,
                layers, batch_size)
        
    model.train(total_time = 40, learning_rate=1e-3)
    
    Shear = np.zeros((300,t_star.shape[0]))
    
    for snap in range(0,t_star.shape[0]):
        
        x1_shear = np.linspace(15,25,100)[:,None]
        x2_shear = np.linspace(25,35,100)[:,None]
        x3_shear = np.linspace(35,55,100)[:,None]
    
        x_shear = np.concatenate([x1_shear,x2_shear,x3_shear], axis=0)
    
        y1_shear = 0.0*x1_shear
        y2_shear = np.sqrt(25.0 - (x2_shear - 30.0)**2)
        y3_shear = 0.0*x3_shear
    
        y_shear = np.concatenate([y1_shear,y2_shear,y3_shear], axis=0)
            
        t_shear = T_star[0,snap] + 0.0*x_shear
        
        eps11_dot_shear, eps12_dot_shear, eps22_dot_shear = model.predict_eps_dot(t_shear, x_shear, y_shear)
        
        nx1_shear = 0.0*x1_shear
        nx2_shear = 6.0 - x2_shear/5.0
        nx3_shear = 0.0*x3_shear
        
        nx_shear = np.concatenate([nx1_shear,nx2_shear,nx3_shear], axis=0)
        
        ny1_shear = -1.0 + 0.0*y1_shear
        ny2_shear = -y2_shear/5.0
        ny3_shear = -1.0 + 0.0*y3_shear
        
        ny_shear = np.concatenate([ny1_shear,ny2_shear,ny3_shear], axis=0)
        
        shear_x = 2.0*(1.0/5.0)*(eps11_dot_shear*nx_shear + eps12_dot_shear*ny_shear)
        shear_y = 2.0*(1.0/5.0)*(eps12_dot_shear*nx_shear + eps22_dot_shear*ny_shear)
        
        shear = np.sqrt(shear_x**2 + shear_y**2)
        
        Shear[:,snap] = shear.flatten()
    
    scipy.io.savemat('../Results/Stenosis2D_Pec_Re_shear_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'Shear':Shear, 'x_shear':x_shear})
    
    # Test Data
    snap = np.array([55])
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
    
    scipy.io.savemat('../Results/Stenosis2D_Pec_Re_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'P_pred':P_pred, 'Pec': model.sess.run(model.Pec), 'Rey': model.sess.run(model.Rey)})
    
    
#    model.sess.run(model.Rey)
#    Out[3]: 4.993976
#
#    model.sess.run(model.Pec)
#    Out[4]: 14.912559
