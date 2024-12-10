import numpy as np

def normalize_landmarks(landmarks, point_idx1=389, point_idx2=227, scale=0.25):
    '''
    Normalize landmarks (extracted by moai tool) to a fixed scale
    left_mouth_point = 389
    right_mouth_point = 227
    '''
    dist = np.linalg.norm(landmarks[:10, point_idx1] - landmarks[:10, point_idx2], axis=1).mean()
    ratio = dist / 256
    landmarks = landmarks * scale / ratio
    
    return landmarks

def get_accelerations(landmarks, idx_stamps, ldmk_idxs):
    lmks_accs = {}
    for ldmk_id in ldmk_idxs:
        # locations
        x = landmarks[:, ldmk_id, 0] # (T, 1)
        y = landmarks[:, ldmk_id, 1]
        # velocities
        v_x = x[1:] - x[:-1]  # (T, 1), velocities
        v_y = y[1:] - y[:-1]
        accs = []
        for i in range(len(idx_stamps)-1):
            st, ed = int(idx_stamps[i]), int(idx_stamps[i+1])
            accs.append([v_x[ed] - v_x[st], v_y[ed] - v_y[st]])
        accs = np.vstack(accs)
        lmks_accs[ldmk_id] = accs
    
    return lmks_accs

def get_period_msi(acces, period=12, inverse=True, bias=0.5):
    a, b = [], []
    for lid in acces.keys():
        a.append(acces[lid][:, 0])  # x coordinate, horizontal locations  # (N, )
        b.append(acces[lid][:, 1])  # y coordinate, vertical locations    # (N, )
    a = np.vstack(a)    # x coords, (M, N-1)
    b = np.vstack(b)    # y coords, (M, N-1)
    msi_x = []
    msi_y = []
    T = period
    for i in range(0, a.shape[1], T//2):
        if inverse:
            msi_x.append((1 / (np.var(a[:, i:i+T], axis=1) + bias)).mean()) # MSI for M keypoints on a sub-sequence
            msi_y.append((1 / (np.var(b[:, i:i+T], axis=1) + bias)).mean()) # MSI for M keypoints on a sub-sequence
        else:
            msi_x.append((np.var(a[:, i:i+T], axis=1)).mean()) # MSI for M keypoints on a sub-sequence
            msi_y.append((np.var(b[:, i:i+T], axis=1)).mean()) # MSI for M keypoints on a sub-sequence
    msi_x = np.array(msi_x)
    msi_y = np.array(msi_y)
    msi = (msi_x + msi_y) / 2
    return msi.mean()

def get_MSI_func(n_frame_landmarks, ldmk_idxs, left_mouth_point, right_mouth_point, scale=0.25, bias=0.5, period=12):
    '''
    Normalize landmarks (extracted by moai tool) to a fixed scale
        left_mouth_point = 389
        right_mouth_point = 227
    
    ldmk_idxs: M target keypoints for measureing motion stability
        landmark indexes for lip: [535, 379, 380, 384, 636, 69, 65, 64, 221] + [226, 219, 224, 74, 538, 539, 385, 258]
        landmark indexes for jaw: [419, 417, 570, 418, 662, 106, 256, 96, 105]
    
    Coefficients:
        scale=0.25, bias=0.5, period=12
    '''
    landmarks = n_frame_landmarks
    landmarks = normalize_landmarks(landmarks, left_mouth_point, right_mouth_point, scale)
    acc_phone1 = get_accelerations(landmarks, np.arange(len(landmarks)-1), ldmk_idxs=ldmk_idxs)
    var_axy = get_period_msi(acc_phone1, period=period, inverse=False, bias=bias)
    var_axy_i = get_period_msi(acc_phone1, period=period, inverse=True, bias=bias)
    return var_axy, var_axy_i

from scipy.signal import savgol_filter

if __name__ == '__main__':
    landmarks_a = np.load('msi/sample.npy')  # shape: (201, 669, 2)
    landmarks_b = savgol_filter(landmarks_a, 5, 2, axis=0) # smooth
    lmk_idxs = [535, 379, 380, 384, 636, 69, 65, 64, 221] + [226, 219, 224, 74, 538, 539, 385, 258]

    _, msi_a = get_MSI_func(landmarks_a, ldmk_idxs=lmk_idxs, left_mouth_point=389, right_mouth_point=227, )
    _, msi_b = get_MSI_func(landmarks_b, ldmk_idxs=lmk_idxs, left_mouth_point=389, right_mouth_point=227, )
    print(msi_a, msi_b)
