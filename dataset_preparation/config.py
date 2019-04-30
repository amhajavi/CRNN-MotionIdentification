import numpy as np
joints = ['root', 'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head', 'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb', 'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb', 'rfemur', 'rtibia', 'rfoot', 'rtoes', 'lfemur', 'ltibia', 'lfoot', 'ltoes']
indexes = np.array([1, 7, 10, 13, 16, 19, 22, 25, 27, 30, 31, 32, 34, 35, 37, 39, 42, 43, 44, 46, 47, 49, 52, 53, 55, 56, 59, 60, 62, 63])-1

locations = dict(zip(joints, zip(indexes,indexes[1:]-indexes[:-1])))
