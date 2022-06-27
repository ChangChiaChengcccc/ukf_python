#!/usr/bin/env python

import rospy
from ukf import UKF
import numpy as np
from numpy.linalg import inv
import math 
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Point

# time variables
time_last = 0
dt = 0

# pre-declare variables
state_dim = 19
measurement_dim = 6
sensor_data = np.zeros(measurement_dim)
RotMat_ned = np.zeros((3,3))
RotMat_enu = np.zeros((3,3))

AlloMat = np.array([
                     [1,   1,   1,   1],
                     [-0.1625, 0.1625, 0.1625, -0.1625],
                     [0.1625, 0.1625, -0.1625, -0.1625],
                     [-2e-2,   2e-2,   -2e-2,   2e-2]    
                   ])
f_cmd_vec = np.zeros(4)
f_M = np.zeros(4)
acc_enu = np.zeros(3)
acc_enu_dyn = np.zeros(3)
W = np.zeros(3)
e3 = np.array([0,0,1])
f1_cmd = 0
f2_cmd = 0
f3_cmd = 0
f4_cmd = 0
acc_dyn = np.zeros(3)
debug = np.zeros(3)

estimate_state_list = Float64MultiArray()
acc_dyn_list = Float64MultiArray()
debug_list = Float64MultiArray()


# Process Noise
q = np.eye(state_dim)
# x,v,a
q[0][0] = 0.001 
q[1][1] = 0.001
q[2][2] = 0.001
q[3][3] = 0.001
q[4][4] = 0.001
q[5][5] = 0.001
q[6][6] = 0.001
q[7][7] = 0.001
q[8][8] = 0.001
# W,dW
q[9][9] = 0.01
q[10][10] = 0.01
q[11][11] = 0.01
q[12][12] = 0.01
q[13][13] = 0.01
q[14][14] = 0.01
# E
q[15][15] = 0.001
q[16][16] = 0.001
q[17][17] = 0.001
q[18][18] = 0.001


# create measurement noise covariance matrices
p_yy_noise = np.eye(measurement_dim)
p_yy_noise[0][0] = 0.001
p_yy_noise[1][1] = 0.001
p_yy_noise[2][2] = 0.001
p_yy_noise[3][3] = 0.0001
p_yy_noise[4][4] = 0.0001
p_yy_noise[5][5] = 0.0001

# create initial state
initial_state = np.zeros(state_dim)


def iterate_x(x, timestep):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    global acc_enu, acc_enu_dyn, f_M_cmd, e3, debug, f1_cmd, f2_cmd, f3_cmd, f4_cmd
    global f_cmd_vec, RotMat_enu, AlloMat, acc_dyn
    m = 1.42
    J = np.diag([0.0117,0.0117,0.0222])
    g = 9.806
    E_vec_from_x = x[15:19]
    E_diag_from_x = np.diag(E_vec_from_x)
    f_cmd_vec = np.array([f1_cmd,f2_cmd,f3_cmd,f4_cmd])
    f_M_cmd = np.dot(AlloMat,f_cmd_vec)
    W_from_x = x[9:12]
    # [f M] = AlloMat*E_diag*f_cmd_vec
    f_M_E = np.dot(AlloMat,np.dot(E_diag_from_x,f_cmd_vec))
    # dynamics
    # a = fRe3/m-ge3
    acc_dyn = f_M_cmd[0]*np.dot(RotMat_enu,e3)/m - g*e3
    
    a_state = f_M_E[0]*np.dot(RotMat_enu,e3)/m - g*e3
    
    # WxJW
    WxJW = np.cross(W_from_x,np.dot(J,W_from_x))
    # M-WxJW
    M_WxJW = f_M_E[1:4] - WxJW
    # dW = inv(J)(M-WxJW)    
    dW_state = np.dot(inv(J),M_WxJW)

    ret = np.zeros(len(x))
    # x,v,a
    # x
    ret[0] = x[0] + x[3] * timestep + 0.5*x[6]*timestep*timestep
    ret[1] = x[1] + x[4] * timestep + 0.5*x[7]*timestep*timestep
    ret[2] = x[2] + x[5] * timestep + 0.5*x[8]*timestep*timestep
    # v
    ret[3] = x[3] + x[6] * timestep
    ret[4] = x[4] + x[7] * timestep
    ret[5] = x[5] + x[8] * timestep
    # a
    ret[6] = a_state[0]
    ret[7] = a_state[1]
    ret[8] = a_state[2]
    # W			
    ret[9] = x[9] + x[12] * timestep
    ret[10] = x[10] + x[13] * timestep
    ret[11] = x[11] + x[14] * timestep
    # dW
    ret[12] = dW_state[0]
    ret[13] = dW_state[1]
    ret[14] = dW_state[2]
    # E
    ret[15] = x[15]
    ret[16] = x[16]
    ret[17] = x[17]
    ret[18] = x[18]
    return ret

def measurement_model(x):
    """
    :param x: states
    """
    # dynamics

    global measurement_dim
    ret = np.zeros(measurement_dim)
    # x,W
    ret[0] = x[0]
    ret[1] = x[1]
    ret[2] = x[2]
    ret[3] = x[9]
    ret[4] = x[10]
    ret[5] = x[11]
    return ret

def pos_enu_cb(data):
    global pos_enu, sensor_data 
    pos_enu = np.array([data.x, data.y, data.z])
    sensor_data[0:3] = np.array([data.x, data.y, data.z])

def gyro_cb(data):
    global W, sensor_data
    W = np.array([data.x, data.y, data.z])
    sensor_data[3:6] = np.array([data.x, data.y, data.z])

def f1_cmd_cb(data):
    global f1_cmd
    f1_cmd = data.data[0]

def f2_cmd_cb(data):
    global f2_cmd
    f2_cmd = data.data[0]

def f3_cmd_cb(data):
    global f3_cmd
    f3_cmd = data.data[0]

def f4_cmd_cb(data):
    global f4_cmd
    f4_cmd = data.data[0]

def RotMat_ned_cb(data):
    global RotMat_ned, RotMat_enu, RotFrame
    RotMat_ned = np.array([
				[data.data[0],data.data[1],data.data[2]],
				[data.data[3],data.data[4],data.data[5]],
				[data.data[6],data.data[7],data.data[8]]
	       		  ])
    RotFrame = np.array([
			[0,1,0],
			[1,0,0],
			[0,0,-1]
		        ])
    RotMat_enu = np.dot(RotFrame,np.dot(RotMat_ned,RotFrame))

def ukf():
    global time_last
    global dt
    dt = rospy.Time.now().to_sec() - time_last
    ukf_module.predict(dt)
    ukf_module.update(measurement_dim, sensor_data, p_yy_noise)
    time_last = rospy.Time.now().to_sec()

    # print('dt:')
    # print(dt)
    # print('rospy.Time.now().to_sec()')
    # print(rospy.Time.now().to_sec())


if __name__ == "__main__":
    try:
        rospy.init_node('UKF')
        state_pub = rospy.Publisher("/offline_ukf_estimated_state", Float64MultiArray, queue_size=10)
        acc_dyn_pub = rospy.Publisher("/offline_ukf_acc_dyn", Float64MultiArray, queue_size=10)
        debug_pub = rospy.Publisher("/offline_ukf_debug", Float64MultiArray, queue_size=10)
        rospy.Subscriber("/pos_enu", Point, pos_enu_cb, queue_size=10)
        rospy.Subscriber("/angular_vel", Point, gyro_cb, queue_size=10)
        rospy.Subscriber("/f1_cmd", Float32MultiArray, f1_cmd_cb, queue_size=10)
        rospy.Subscriber("/f2_cmd", Float32MultiArray, f2_cmd_cb, queue_size=10)
        rospy.Subscriber("/f3_cmd", Float32MultiArray, f3_cmd_cb, queue_size=10)
        rospy.Subscriber("/f4_cmd", Float32MultiArray, f4_cmd_cb, queue_size=10)
        rospy.Subscriber("/RotMat_ned", Float32MultiArray, RotMat_ned_cb, queue_size=10)

        # pass all the parameters into the UKF!
        # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
        #def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function, measurement_model):
        ukf_module = UKF(state_dim, q, initial_state, 0.001*np.eye(state_dim), 0.001, 0.0, 2.0, iterate_x, measurement_model)
        rate = rospy.Rate(40)
        print("start ukf model!")
        while not rospy.is_shutdown():         
            ukf()
            estimate_state = ukf_module.get_state()
            estimate_state_list.data = list(estimate_state)
            state_pub.publish(estimate_state_list)
            
            acc_dyn_list.data = list(acc_dyn)
            acc_dyn_pub.publish(acc_dyn_list)

            debug_list.data = list(debug)
            debug_pub.publish(debug_list)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
