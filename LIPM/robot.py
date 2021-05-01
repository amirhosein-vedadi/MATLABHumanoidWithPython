import numpy as np

def find_initial_conditions(step_length, dy_mid, x0, z_model, g, sample_time):
    """"""
    time_cons = np.sqrt(z_model / g)
    y_mid = 0
    orbital_energy = -g/(2 * z_model) * y_mid ** 2 + 0.5 * dy_mid **2
    y0 = -step_length / 2
    dy0 = np.sqrt(2 * (orbital_energy + g / (2 * z_model) * y0 ** 2))
    t_single_support = 2 * np.arcsinh(step_length/2/(time_cons *dy_mid)) * time_cons
    t_single_support = np.floor(t_single_support / sample_time) * sample_time
    tf = t_single_support / 2
    dx0 = -x0 / time_cons * np.sinh(tf / time_cons) / np.cosh(tf / time_cons)
    
    return [dx0, y0, dy0, t_single_support]
    
class ModelVars():
    
    def __init__(self, g, sample_time, z_model, t_single_support, Ad, Bd, Cd, Dd):
        
        self.g = g
        self.sample_time = sample_time
        self.z_model = z_model
        self.t_single_support = t_single_support
        self.Ad = Ad
        self.Bd = Bd
        self.Cd = Cd
        self.Dd = Dd

class SimStates():
    
    def __init__(self, body_pos, foot_pos, time_vec):
    
        self.body_pos = body_pos
        self.foot_pos = foot_pos
        self.time_vec = time_vec
    
class StepInfo():
    def __init__(self, index, state, time_vec, mode, footplant, swing=None):
        self.index = index
        self.state = state
        self.time_vec = time_vec
        self.mode = mode
        self.footplant = footplant
        self.swing = swing

def stance_sim(fhold_x, fhold_y, state0, u0, model_vars, sim_states):
    
    ts = model_vars.sample_time
    Ad = model_vars.Ad
    Bd = model_vars.Bd
    
    init_t = sim_states.time_vec[-1] + ts
    final_t = init_t + model_vars.t_single_support
    time_vec = np.arange(init_t, final_t, ts)
    n_steps = np.size(time_vec)
    states = np.zeros((state0.shape[0], n_steps))
    states[:, 0] = state0.reshape(-1)
    
    for idx in range(n_steps - 1):
        states[:, idx + 1] = np.matmul(Ad, states[:, idx]) + np.matmul(Bd, u0).reshape(-1)
    
    sim_states.time_vec = np.append(sim_states.time_vec, time_vec)
    new_states = np.array([[states[0, 1:]+fhold_x], [states[2, 1:]+fhold_y], model_vars.z_model * np.ones((1, n_steps - 1))])
    new_states = new_states.reshape(3, -1)
    sim_states.body_pos = np.append(sim_states.body_pos, new_states, 1)
    
    return [states, time_vec, sim_states]

def change_leg(state1, sim_states):
    
    xf  = state1[0]
    dxf = state1[1]
    yf  = state1[2]
    dyf = state1[3]
    
    x0 = -xf
    y0 = -yf 
    dx0 = dxf 
    dy0 = dyf
    state0 = np.array([[x0], [dx0], [y0], [dy0]])
    
    fhold_x = sim_states.body_pos[0, -1] - x0
    fhold_y = sim_states.body_pos[1, -1] - y0
    
    return [state0, fhold_x, fhold_y]
    
class FootInfo():
    def __init__(self, time_vec, foot_left, foot_right):
        self.time_vec = time_vec
        self.foot_left = foot_left
        self.foot_right = foot_right
        
    def set_joints_left(self, joints_left):
        self.joints_left = joints_left
        
    def set_joints_right(self, joints_right):
        self.joints_right = joints_right
        
    def set_trans_mat_left(self, trans_mat_left):
        self.trans_mat_left = trans_mat_left
        
    def set_trans_mat_right(self, trans_mat_right):
        self.trans_mat_right = trans_mat_right
