import numpy as np

def inv_kin_body2foot(tform, is_left):

    l1 = -0.12
    if is_left:
        l1 *= -1
    l2 = 0
    l3 = 0.4
    l4 = 0.38
    l5 = 0

    tform[0, 3] *= -1
    tform -= np.array([[0, 0, 0, l1], [0, 0, 0, 0], [0, 0, 0, -l2], [0, 0, 0, 0]])
    R = tform[0:3, 0:3]
    p = tform[0:3, 3]
    Rp = R.T
    n = Rp[:, 0] 
    s = Rp[:, 1] 
    a = Rp[:, 2]
    p = -np.matmul(Rp, p)

    cos4 = ((p[0]+l5)**2 + p[1]**2 + p[2]**2 - l3**2 - l4**2)/(2*l3*l4)
    temp = 1 - cos4 ** 2
    if temp < 0:
        temp = 0
        print('Waning: Unable to reach desired end-effector position/orientation')
    
    th4 = np.arctan2(np.sqrt(temp), cos4)

    temp = (p[0]+l5)**2 + p[1]**2
    if temp < 0:
        temp = 0
        print('Waning: Unable to reach desired end-effector position/orientation')

    th5 = np.arctan2(-p[2], np.sqrt(temp)) - np.arctan2(np.sin(th4)*l3, np.cos(th4)*l3+l4)
    th6 = np.arctan2(p[1], -p[0]-l5)
    temp = 1 - (np.sin(th6)*a[0]+np.cos(th6)*a[2])**2
    if temp < 0:
        temp = 0
        print('Waning: Unable to reach desired end-effector position/orientation')

    th2 = np.arctan2(-np.sqrt(temp), np.sin(th6)*a[0]+np.cos(th6)*a[1])
    th2 += np.pi / 2
    th1 = np.arctan2(-np.sin(th6)*s[0]-np.cos(th6)*s[1],-np.sin(th6)*n[0]-np.cos(th6)*n[2])

    th345 = np.arctan2(a[2], np.cos(th6)*a[0] - np.sin(th6)*a[1])
    th345 -= np.pi
    th3 = th345 - th4 - th5

    return [th1, th2, th3, th4, th5, th6]

