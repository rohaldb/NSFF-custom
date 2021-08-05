import numpy as np
import sys
sys.path.insert(1, '/content/Neural-Scene-Flow-Fields/nsff_exp')
from load_llff import _load_data, recenter_poses
from pdb import set_trace
import imageio
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--final_height", type=int, default=288,
                        help='training image height, default is 512x288')

    return parser

def get_poses_and_hwf(datadir, final_height, bd_factor):

    poses, bds = _load_data(datadir, start_frame=0, end_frame=None, height=final_height, load_imgs=False)

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :],
                            -poses[:, 0:1, :],
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc

    poses = recenter_poses(poses).astype(np.float32)

    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]

    return poses, hwf

# inverts a pose matrix to obtain an extrinsic matrix
def convert_pose_to_extrinsic(pose):
  R_w2c = pose[:3, :3].T
  t_w2c = -R_w2c @ pose[:3, 3:]
  return np.hstack((R_w2c, t_w2c))

# applies a transformation matrix to a point in homogeneous space
def apply_transformation(matrix, points):
  hom_point = np.vstack((points, np.ones(points.shape[1])))
  hom_res = matrix @ hom_point
  return hom_res[:3]/hom_res[-1]

def convert_hwf_to_extrinsic(hwf):
    K = np.eye(3)
    K[0, 0] = hwf[2]
    K[0, 2] = hwf[1] / 2.
    K[1, 1] = hwf[2]
    K[1, 2] = hwf[0] / 2.
    return K

def homogeneous(vec):
  return np.concatenate((vec, [1]))

# computes the scene flow induced by camera ego motion
def compute_s_cam(h, w, K, depth, cOne2w, w2cTwo):

    p_matrix = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            p_matrix[y, x] = [y, x, 1]

    z = p_matrix.reshape(-1, 3).T
    # P is in cam1 space
    P_c1 = (np.linalg.inv(K) @ z) * depth.flatten()
    # convert to cam2's space
    P_w = apply_transformation(cOne2w, P_c1)
    P_c2 = apply_transformation(w2cTwo, P_w)
    # obtain ego-motion induced sf due to camera coordinate differences
    S_cam = P_c2.T.reshape(p_matrix.shape) - P_c1.T.reshape(p_matrix.shape)

    return S_cam

def subtract_sf_due_to_ego_motion(datadir, poses, K):

    # ---- forward scene flow ----

    fw_dir = datadir + "/sf_corrected"
    bw_dir = datadir + "/sf_bw_corrected"
    if not os.path.exists(fw_dir):
        os.makedirs(fw_dir)
        print("{} has been created.".format(fw_dir))
    if not os.path.exists(bw_dir):
        os.makedirs(bw_dir)
        print("{} has been created.".format(bw_dir))

    #sf in fwd direction
    for i in tqdm(range(3, len(poses)-1)):
        sf_fw_dir = datadir + "/sf/%05d.npy"%i
        sf_fw = np.load(sf_fw_dir)
        sf_bw_dir = datadir + "/sf_bw/%05d.npy" % i
        sf_bw = np.load(sf_bw_dir)
        # move mono-sf from [x,-y,-z] to nsff's [x, y, z]
        sf_fw *= [1, -1, -1]
        sf_bw *= [1, -1, -1]
        depth_dir = datadir + "/depth_0/%05d.npy"%i
        depth = np.load(depth_dir)
        mask_dir = datadir + "/motion_masks/%05d.png"%i
        mask = imageio.imread(mask_dir)
        # print("loaded from", sf_fw_dir, sf_bw_dir, depth_dir, mask_dir)
        mask = mask < 1e-3

        # get matricies and compute sf due to camera
        cZero2w = np.vstack((poses[i-1], np.array([0, 0, 0, 1])))
        cOne2w = np.vstack((poses[i], np.array([0,0,0,1])))
        cTwo2w = np.vstack((poses[i+1], np.array([0,0,0,1])))
        w2cTwo = convert_pose_to_extrinsic(cTwo2w)
        w2cZero = convert_pose_to_extrinsic(cZero2w)
        s_cam_fw = compute_s_cam(sf_fw.shape[0], sf_fw.shape[1], K, depth, cOne2w, w2cTwo)
        s_cam_bw = compute_s_cam(sf_fw.shape[0], sf_fw.shape[1], K, depth, cOne2w, w2cZero)


        # compute scaling factors and save
        x = s_cam_fw.reshape(-1, 3)[mask.reshape(-1), :]
        y = sf_fw.reshape(-1, 3)[mask.reshape(-1), :]
        reg = LinearRegression().fit(x, y)
        corrected_sf_fw = sf_fw - reg.predict(s_cam_fw.reshape(-1, 3)).reshape(288, 512, 3)
        np.save(fw_dir + "/%05d.npy"%i, corrected_sf_fw)

        x = s_cam_bw.reshape(-1, 3)[mask.reshape(-1), :]
        y = sf_bw.reshape(-1, 3)[mask.reshape(-1), :]
        reg = LinearRegression().fit(x, y)
        corrected_sf_bw = sf_bw - reg.predict(s_cam_bw.reshape(-1, 3)).reshape(288, 512, 3)
        np.save(bw_dir + "/%05d.npy" % i, corrected_sf_bw)



if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    bd_factor = 0.9
    poses, hwf = get_poses_and_hwf(args.datadir, args.final_height, bd_factor)
    K = convert_hwf_to_extrinsic(hwf)
    subtract_sf_due_to_ego_motion(args.datadir, poses, K)