#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import numpy as np

from draw import Draw_MPC_point_stabilization_v1, draw_gt, draw_gt_measurements, draw_gtmeas_noisemeas, draw_gt_mhe_measurements

def shift_movement(T, t0, x0, u, x_n, f):
    f_value = f(x0, u[0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return t, st, u_end, x_n

def prediction_state(x0, u, T, N):
    # define predition horizon function
    states = np.zeros((N+1, 3))
    states[0, :] = x0
    for i in range(N):
        states[i+1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * T
        states[i+1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * T
        states[i+1, 2] = states[i, 2] + u[i, 1] * T
    return states


if __name__ == '__main__':
    T = 0.2
    N = 10
    rob_diam = 0.3 # [m]
    v_max = 0.6
    omega_max = np.pi/4.0

    opti = ca.Opti()
    # control variables, linear velocity v and angle velocity omega
    n_controls = 2
    opt_controls = opti.variable(N, n_controls)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]
    n_states = 3
    opt_states = opti.variable(N+1, n_states)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    # parameters
    opt_x0 = opti.parameter(3)
    opt_xs = opti.parameter(3)
    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])
    f_np = lambda x_, u_: np.array([u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :]==x_next)

    ## define the cost function
    ### some addition parameters
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, .1]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :]-opt_xs.T), Q, (opt_states[i, :]-opt_xs.T).T]) + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])

    opti.minimize(obj)

    #### boundrary and control conditions
    opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(opti.bounded(-2.0, y, 2.0))
    opti.subject_to(opti.bounded(-np.inf, theta, np.inf))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    opts_setting = {'ipopt.max_iter':2000, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    opti.solver('ipopt', opts_setting)




    t0 = 0
    init_state = np.array([0.1, 0.1, 0.0])
    final_state = np.array([1.5, 1.5, 0.0])
    opti.set_value(opt_xs, final_state)
    next_controls = np.zeros((N, 2))
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 3))
    x_c = [] # contains for the history of the state
    u_c = []
    # t_c = [t0] # for the time
    t_c = []
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0

    while(np.linalg.norm(current_state-final_state)>0.05 and mpciter-sim_time/T<0.0  ):
        opti.set_value(opt_x0, current_state)
        opti.set_initial(opt_controls, next_controls.reshape(N, n_controls))# (N, 2)
        opti.set_initial(opt_states, next_states) # (N+1, 3)
        ## solve the problem once again
        sol = opti.solve()
        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)
        u_c.append(u_res[0, :])
        t_c.append(t0)
        x_c.append(x_m)
        t0, current_state, u0, next_states = shift_movement(T, t0, current_state, u_res, x_m, f_np)
        xx.append(current_state)
        mpciter = mpciter + 1

    xx_np = np.array(xx)
    # draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=init_state, target_state=final_state,robot_states=xx)
    # draw_gt(t_c, xx_np)

    # synthesize the measurements
    con_cov = np.diag([0.005, np.deg2rad(2)])**2
    meas_cov = np.diag([0.1, np.deg2rad(2)])**2

    r = []
    alpha = []
    for i in range(len(xx)):
        r.append(np.sqrt(xx_np[i, 0]**2+xx_np[i, 1]**2) + np.sqrt(meas_cov[0, 0])*np.random.randn())
        alpha.append(np.arctan(xx_np[i, 1]/xx_np[i, 0]) + np.sqrt(meas_cov[1, 1])*np.random.randn())
    y_measurements = np.concatenate((np.array(r).reshape(-1, 1), np.array(alpha).reshape(-1, 1)), axis=1)

    # draw_gt_measurements(t_c, xx_np, y_measurements)
    # draw_gtmeas_noisemeas(t_c, xx_np, y_measurements)


    ## MHE
    ### now we only know the control u_c and y_measurements
    mhe_opt = ca.Opti() # new optimization class
    u_c_np = np.array(u_c)
    T_mhe = 0.2
    N_MHE = y_measurements.shape[0] - 1 # estimation horizon
    print("MHE horizon {}".format(N_MHE))
    ### using the same model and states （states, f, next_controls)
    mhe_states = mhe_opt.variable(N_MHE+1, n_states)
    mhe_x = mhe_states[:, 0]
    mhe_y = mhe_states[:, 1]
    mhe_theta = mhe_states[:, 2]
    mhe_controls = mhe_opt.variable(N_MHE, n_controls)
    mhe_v = mhe_controls[:, 0]
    mhe_omega = mhe_controls[:, 1]
    mhe_mes_ref = mhe_opt.parameter(N_MHE+1, 2)
    mhe_u_ref = mhe_opt.parameter(N_MHE, 2)
    ## measurement model
    f_m = lambda x, y: ca.vertcat(*[np.sqrt(x**2+y**2), np.arctan(y/x)])
    V_mat = np.linalg.inv(np.sqrt(meas_cov))
    W_mat = np.linalg.inv(np.sqrt(con_cov))

    obj_mhe = 0
    for i in range(N_MHE+1):
        h_x = f_m(mhe_states[i, 0], mhe_states[i, 1])
        temp_diff_ = mhe_mes_ref[i, :] - h_x.T
        obj_mhe = obj_mhe + ca.mtimes([temp_diff_, V_mat, temp_diff_.T])

    for i in range(N_MHE):
        temp_diff_ = mhe_u_ref[i, :] - mhe_controls[i, :]
        obj_mhe = obj_mhe + ca.mtimes([temp_diff_, W_mat, temp_diff_.T])

    mhe_opt.minimize(obj_mhe)

    ## multiple shooting constraints
    for i in range(N_MHE):
        x_next_ = mhe_states[i, :] + f(mhe_states[i, :], mhe_controls[i, :]).T*T
        mhe_opt.subject_to(mhe_states[i+1, :]==x_next_)

    #### boundrary and control conditions
    mhe_opt.subject_to(mhe_opt.bounded(-2.0, mhe_x, 2.0))
    mhe_opt.subject_to(mhe_opt.bounded(-2.0, mhe_y, 2.0))
    mhe_opt.subject_to(mhe_opt.bounded(-np.inf, mhe_theta, np.inf))
    mhe_opt.subject_to(mhe_opt.bounded(-v_max, mhe_v, v_max))
    mhe_opt.subject_to(mhe_opt.bounded(-omega_max, mhe_omega, omega_max))
    mhe_opt.solver('ipopt', opts_setting)

    ## MHE simulation
    mhe_opt.set_value(mhe_mes_ref, y_measurements)
    mhe_opt.set_value(mhe_u_ref, np.array(u_c[:-1]))
    ### initial states from the measured range and bearning
    init_state_m = np.zeros((N_MHE+1, n_states))
    for i in range(N_MHE+1):
        init_state_m[i] = np.array([y_measurements[i, 0]*np.cos(y_measurements[i, 1]),
                                    y_measurements[i, 0]*np.sin(y_measurements[i, 1]),
                                    0.0])
    mhe_opt.set_initial(mhe_states, init_state_m)
    mhe_opt.set_initial(mhe_controls, np.array(u_c[:N_MHE]))# (N, 2)
    ## solve the MHE
    sol = mhe_opt.solve()
    ## obtain the control input
    u_sol = sol.value(mhe_controls)
    state_sol = sol.value(mhe_states)
    draw_gt_mhe_measurements(t_c, xx_np, y_measurements, state_sol)
