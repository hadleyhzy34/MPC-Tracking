import math 
import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time
from matplotlib import pyplot as plt
from draw import Draw_MPC_tracking
from cubic_spline_planner import Spline2D

def forward(T, N, r = 0.15):
    '''
    description: mpc traj follower solver
    args:
        T: sampling time, second
        N: prediction horizon, number of steps
        r: robot radius, default: 0.15m
    return: mpc solver
    '''
    
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.size()[0]

    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    eps = 0.0001
    # rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta))
    rhs = ca.vertcat(v*(ca.sin(theta+(omega+eps)*T)-ca.sin(theta))/(omega+eps),
                      -v*(ca.cos(theta+(omega+eps)*T)-ca.cos(theta))/(omega+eps))
    rhs = ca.vertcat(rhs, omega*T)
    # rhs = ca.vertcat(rhs, omega)

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N+1)
    # U_ref = ca.SX.sym('U_ref', n_controls, N)
    X_ref = ca.SX.sym('X_ref', n_states, N+1)

    ### define
    Q = np.array([[10.0, 0.0, 0.0],[0.0, 10.0, 0.0],[0.0, 0.0, 0.0]])
    # Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])

    R = np.array([[0.0, 0.0], [0.0, 0.0]])
    #### cost function
    obj = 0 #### cost
    g = [] # equal constrains
    g.append(X[:, 0] - X_ref[:, 0])
    # import ipdb;ipdb.set_trace()
    for i in range(N+1):
        # state_error_ = X[:, i] - X_ref[:, i]
        state_error_ = X[:, i] - X_ref[:, i]
        # control_error_ = U[:, i] - U_ref[:, i]
        # obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) + ca.mtimes([control_error_.T, R, control_error_])
        obj = obj + ca.mtimes([state_error_.T, Q, state_error_])
        if i != N:
            # x_next_ = f(X[:, i], U[:, i])*T +X[:, i]
            x_next_ = f(X[:, i], U[:, i]) +X[:, i]
            g.append(X[:, i+1]-x_next_)
    
    # import ipdb;ipdb.set_trace()
    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
    # opt_params = ca.vertcat(ca.reshape(U_ref, -1, 1), ca.reshape(X_ref, -1, 1))
    # opt_params = ca.vertcat(ca.reshape(X_ref, -1, 1))
    opt_params = ca.reshape(X_ref, -1 ,1)

    nlp_prob = {'f': obj, 'x': opt_variables, 'p':opt_params, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 
                    'ipopt.print_level':0, 
                    'print_time':0, 
                    'ipopt.acceptable_tol':1e-10, 
                    'ipopt.acceptable_obj_change_tol':1e-8}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
    
    return solver

def mpc(solver, ref_traj, T, N, v_max, omega_max):
    '''
    description: mpc trajectory application
    args:
        v_max: maximum linear velocity
        omega_max: maximum angular velocity
        solver: mpc solver
        ref_traj: reference trajectory, shape(3,N+1)
    return:
        u_0:[linear_velocity, angular_velocity]
    '''
    solver = forward(T, N)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []
    
    
    for _ in range(N):
        lbx.append(0.01)
        lbx.append(-omega_max)
        ubx.append(v_max)
        ubx.append(omega_max)
    for _ in range(N+1): # note that this is different with the method using structure
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)
    
    
    '''
    for _ in range(N):
        lbx = lbx + [0, -omega_max, -np.inf, -np.inf, -np.inf]
        ubx = ubx + [v_max, omega_max, np.inf, np.inf, np.inf]

    lbx = lbx + [-np.inf, -np.inf, -np.inf]
    ubx = ubx + [np.inf, np.inf, np.inf]
    '''

    # initial states initialization
    states = np.zeros((3,N+1))
    controls = np.zeros((2,N))
    init_states = np.concatenate((controls.T.reshape(-1,1), states.T.reshape(-1,1)))

    # ref trajectory initialization
    # ref_controls = np.zeros((2,N))
    # c_p = np.concatenate((ref_controls.reshape(-1,1), ref_traj.reshape(-1,1)))
    c_p = ref_traj.T.reshape(-1, 1)
    # c_p = ref_traj.reshape(-1, 1)

    # solver computation
    # import ipdb;ipdb.set_trace()
    res = solver(x0 = init_states, p = c_p, lbg = lbg, lbx = lbx, ubg = ubg, ubx = ubx)
    estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
    # import ipdb;ipdb.set_trace()
    u0 = estimated_opt[:int(2*N)].reshape(N, 2).T # (n_controls, N)
    x_m = estimated_opt[int(2*N):].reshape(N+1, 3).T #(n_states, N+1)
    # import ipdb;ipdb.set_trace()
    return u0[:,0], x_m
   
def mpc_api_(ref_path, N, T, v_max, omega_max):
    '''
    description: mpc path tracking
    args:
        ref_path:(5||10||15, 2)
        N:number of horizon
        T:sampling time
        v_max:linear velocity
        omega_max:angular velocity
    return:
        U0:[linear_velocity, angular_velocity]
    '''
    # import ipdb;ipdb.set_trace()
    x = ref_path[:,0]
    y = ref_path[:,1]

    ds = T * v_max/2
    
    sp = Spline2D(x,y)
    s = np.arange(0, sp.s[-1], ds)

    '''
    ref_traj = np.zeros((3,N+1))
    for i,i_s in enumerate(s):                                                               
        if i > N:
            break
        ix, iy = sp.calc_position(i_s)                                          
        ref_traj[0,i] = ix
        ref_traj[1,i] = iy
        # print(i)
    '''

    ref_traj = []
    for i, i_s in enumerate(s):
        ix, iy = sp.calc_position(i_s)
        ref_traj.append([ix,iy,0.0])

    # import ipdb;ipdb.set_trace()
    ref_traj = np.array(ref_traj)
    ref_traj = ref_traj.T

    # import ipdb;ipdb.set_trace()
    solver = forward(T, N)
    u0,x_m = mpc(solver, ref_traj[:,:N+1], T, N, v_max, omega_max)
    # import ipdb;ipdb.set_trace()
    
    obj = 0 #### cost
    # Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    Q = np.array([[10.0, 0.0, 0.0],[0.0, 10.0, 0.0],[0.0, 0.0, 0.0]])
    

    # import ipdb;ipdb.set_trace()
    for i in range(N):
        state_error_ = x_m[:, i] - ref_traj[:, i]
        obj = obj + state_error_.T.dot(Q).dot(state_error_)
        # obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) 
    print("obj is: ", obj)

    # visualization
    # plt.plot(ref_traj[0,:], ref_traj[1,:], "-r", label="input")
    plt.scatter(ref_traj[0,:], ref_traj[1,:], s=(0.5)**2)
    plt.plot(x_m[0,:], x_m[1,:], "xb", label="mpc")

    for i in range(N+1):
        plt.text(ref_traj[0,i], ref_traj[1,i], str(i))
        plt.text(x_m[0,i], x_m[1,i], str(i), color="red")

    plt.grid(True)
    plt.show()
    plt.close()
    return u0

def mpc_api(ref_traj, solver, N, T, v_max, omega_max):
    '''
    args:
    description: mpc path tracking
    args:
        ref_path:shape(5||10||15, 2)
        solver: nlp solver
        N:number of horizon
        T:sampling time
        v_max:linear velocity
        omega_max:angular velocity
    return:
        U0:[linear_velocity, angular_velocity]
    '''

    ####################spline reference trajectory################
    # import ipdb;ipdb.set_trace()
    x = ref_path[:,0]
    y = ref_path[:,1]

    ds = T * v_max/2
    
    sp = Spline2D(x,y)
    s = np.arange(0, sp.s[-1], ds)
    
    ref_traj = []
    for i, i_s in enumerate(s):
        ix, iy = sp.calc_position(i_s)
        ref_traj.append([ix,iy,0.0])

    # import ipdb;ipdb.set_trace()
    ref_traj = np.array(ref_traj)
    ref_traj = ref_traj.T

    # mpc controller
    u0,x_m = mpc(solver, ref_traj[:,:N+1], T, N, v_max, omega_max)
    
    ####################visualization########################
    obj = 0 #### cost
    # Q = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    Q = np.array([[10.0, 0.0, 0.0],[0.0, 10.0, 0.0],[0.0, 0.0, 0.0]])
    

    # import ipdb;ipdb.set_trace()
    for i in range(N):
        state_error_ = x_m[:, i] - ref_traj[:, i]
        obj = obj + state_error_.T.dot(Q).dot(state_error_)
        # obj = obj + ca.mtimes([state_error_.T, Q, state_error_]) 
    print("obj is: ", obj)

    # visualization
    # plt.plot(ref_traj[0,:], ref_traj[1,:], "-r", label="input")
    plt.scatter(ref_traj[0,:], ref_traj[1,:], s=(0.5)**2)
    plt.plot(x_m[0,:], x_m[1,:], "xb", label="mpc")
    plt.legend(['reference trajectory', 'mpc predicted states'], loc = "lower right")
    plt.xlabel("x axes frame")
    plt.ylabel("y axes frame")

    '''
    for i in range(N+1):
        plt.text(ref_traj[0,i], ref_traj[1,i], str(i))
        plt.text(x_m[0,i], x_m[1,i], str(i), color="red")
    '''

    plt.grid(True)
    plt.show()
    plt.close()
    return u0
 
if __name__ == '__main__':
    print('spline 2D test')
    ref_path = np.array([[0.0, 0.0],
                         [0.2, 0.1],
                         [0.3, 0.3],
                         [0.5, 0.4],
                         [0.6, 0.6],
                         [0.8, 0.7]])

    T = 0.1
    N = 50
    v_max = 0.15
    omega_max = 0.5
    
    # declare solver only needed once, inside agent class or sth similar
    solver = forward(T, N)
    # mpc controller, return [linear_velocity, angular_velocity]
    res = mpc_api(ref_path, solver, N, T, v_max, omega_max)
    print(res)
