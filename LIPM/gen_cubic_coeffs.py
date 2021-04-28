import numpy as np

def gen_cubic_coeffs(pos_pts, vel_pts, final_time):
    
    x0 = pos_pts[0]
    dx0 = vel_pts[0]
    xf = pos_pts[1]
    dxf = vel_pts[1]
        
    coeff_vec = np.array([x0, dx0, 0, 0])
    t_mat0 = np.array([[1, final_time], [0, 1]])
    B = np.array([xf, dxf]) - t_mat0.dot(coeff_vec[0:2])
    inv_t_matf = np.array([[3 / final_time**2, -1 / final_time], [-2 / final_time**3, 1 / final_time**2]])
    coeff_vec[2:4] = inv_t_matf.dot(B)
    
    return coeff_vec