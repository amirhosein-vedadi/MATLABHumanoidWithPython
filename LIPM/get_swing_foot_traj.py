import numpy as np

from cubic_poly_traj import *

def get_swing_foot_traj(foot_pos0, foot_pos1, swing_height, time_stp0, time_stpf, sample_time):
    
    time_stpf += 0.001 # our time vector must includes time_stpf 
    way_pts_xy = np.append(foot_pos0[0:2].reshape(2,1), foot_pos1[0:2].reshape(2,1), 1)
    time_pts_xy = np.array([time_stp0, time_stpf])
    time_vec_xy = np.arange(time_stp0, time_stpf, sample_time)
    [xy, xyd, xydd] = cubic_poly_traj(way_pts_xy, time_pts_xy, time_vec_xy)
    
    way_pts_z = [foot_pos0[2], foot_pos0[2]+swing_height, foot_pos1[2]]
    time_stp_mid = (time_stp0 + time_stpf) / 2
    time_pts_z = np.array([time_stp0, time_stp_mid, time_stpf])
    time_vec_z = np.arange(time_stp0, time_stpf, sample_time)
    [z, zd, zdd] = cubic_poly_traj(way_pts_z, time_pts_z, time_vec_z)
    
    q = np.append(xy, z).reshape(3, -1)
    qd = np.append(xyd, zd).reshape(3, -1)
    qdd = np.append(xydd, zdd).reshape(3, -1)
    
    return [q, qd, qdd]