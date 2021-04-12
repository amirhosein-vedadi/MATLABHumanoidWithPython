import numpy as np

def cubic_poly_traj (way_points, time_points, t, vel_points=None):
    
    way_points = np.asarray(way_points)
    time_points = np.asarray(time_points)
    t = np.asarray(t).reshape(-1)
    
    if len(way_points.shape) == 1:
        n = 1
        p = way_points.shape[0]
    else:
        n = way_points.shape[0]
        p = way_points.shape[1]
        
    sample_time = t[1] - t[0]
    
    if vel_points is None:
        if n == 1:
            vel_points = np.zeros(p)
        else:
            vel_points = np.zeros((n, p))
    else:
        vel_points = np.asarray(vel_points)
    
    q = np.zeros((n, t.shape[0]))
    qd = np.zeros((n, t.shape[0]))
    qdd = np.zeros((n, t.shape[0]))
    
    coef_dim = 4
    coef_mat = np.zeros(((p - 1) * n, coef_dim))
    
    for i in range(p-1):
        
        final_time = time_points[i+1] - time_points[i]
        
        for j in range(n):
            
            ridx = i * n + j
            if n == 1:
                coef_mat[ridx, :] = gen_cubic_coeffs(way_points[i:i+2], vel_points[i:i+2], final_time)
            else:
                coef_mat[ridx, :] = gen_cubic_coeffs(way_points[j, i:i+2], vel_points[j, i:i+2], final_time)
            
    for i in range(t.shape[0]):
        
        for j in range(p-1):
            
            if (t[i] < time_points[j + 1] and t[i] >= time_points[j]) or (t[i] == time_points[-1]):
                
                q[:, i] = coef_mat[j * n:(j+1) * n, :].dot(np.array([1, t[i]-time_points[j], (t[i]-time_points[j])**2, (t[i]-time_points[j])**3]))
                qd[:, i] = coef_mat[j * n:(j+1) * n, 1:4].dot(np.array([1, 2*(t[i]-time_points[j]), 3*(t[i]-time_points[j])**2]))
                qdd[:, i] = coef_mat[j * n:(j+1) * n, 2:4].dot(np.array([2, 6*(t[i]-time_points[j])]))
            else:
                
                pass
    return [q, qd, qdd]

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