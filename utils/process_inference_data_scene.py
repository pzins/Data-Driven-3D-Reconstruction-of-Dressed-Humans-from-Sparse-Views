import numpy as np
import json
import cv2
import os
import sys
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import trimesh

def name_from_yaw(yaw, yaw_pitch):
    if yaw_pitch is None:
        return str(yaw)
    return str(yaw) + "_" + str(yaw_pitch[str(yaw)]) + "_00"

def convert_calib_to_dict(calib):
    res = {}
    for i in ["pos", "rot", "K", "Rt", 'norm', 'proj', 'sh', 'ortho_ratio', 'scale', 'center', 'R']:
        res[i] = calib.item().get(i)
    return res



# PARAMETERS
# ==============================================
folder = ""
subject = "" #sys.argv[1]
projection = "persp"

crop_img_size = 512
orig_img_size = 2048

debug = True
debug = False

suffix = "_orig"
suffix = ""
cams = [0, 90, 180, 270]
# ==============================================
if subject == "" or folder == "":
    print("Parameters should be specified")
    exit(0)


points = []
calibs = []

with open(os.path.join(folder, "RENDER", subject, "yaw_pitch.json"), 'r') as f:
    yaw_pitch = json.load(f)

os.makedirs(os.path.join(folder, "INFERENCE/MASK", subject), exist_ok=True)
os.makedirs(os.path.join(folder, "INFERENCE/RENDER", subject), exist_ok=True)
os.makedirs(os.path.join(folder, "INFERENCE/PARAM", subject), exist_ok=True)

for cam in cams:
    render_filename = os.path.join(folder, "RENDER" + suffix, subject, name_from_yaw(cam, yaw_pitch) + ".jpg")
    if os.path.isfile(render_filename):
        image = Image.open(render_filename)
    else:
        image = Image.open(render_filename[:-4] + ".png")
    mask_image = Image.open(os.path.join(folder, "MASK" + suffix, subject, name_from_yaw(cam, yaw_pitch) + ".png"))
    
    # Use keypoints to get uppper left and bottom right points
    with open(os.path.join(folder, "KEYPOINTS", subject, name_from_yaw(cam, yaw_pitch) + "_keypoints.json")) as f:
        data = json.load(f)
    
    if len(data["people"]) == 0:
        continue
    pos = data["people"][0]["pose_keypoints_2d"][8*3:9*3-1] # PELVIS
    # pos : [position horiz, position vert]
    
    points.append(np.array(pos))

    use_all_joints = True
    
    if use_all_joints:
        # Bouding box using all joints
        keypoints = data["people"][0]["pose_keypoints_2d"]
        keypoints = np.array(keypoints).reshape(-1, 3)
        keypoints = keypoints[keypoints[:,2] > 0.1][:,:2]
        bounds = [np.floor(keypoints.max(0)).tolist(), np.floor(keypoints.min(0)).tolist()]
        dh = crop_img_size - (bounds[0][1] - bounds[1][1])
        dh1 = dh//2
        dh2 = dh - dh1 

        dw = crop_img_size - (bounds[0][0] - bounds[1][0])
        dw1 = dw//2
        dw2 = dw - dw1
        render = image.crop((bounds[1][0] - dw1, bounds[1][1] - dh1, bounds[0][0] + dw2, bounds[0][1] + dh2))
        mask_render = mask_image.crop((bounds[1][0] - dw1, bounds[1][1] - dh1, bounds[0][0] + dw2, bounds[0][1] + dh2))
        offset = np.array([bounds[1][0] - dw1, bounds[1][1] - dh1])
    else:
        # Bounding box using a single joint
        # crop(left, up, right, bottom)
        left = pos[0] - crop_img_size/2
        up = pos[1] - crop_img_size/2
        right = pos[0] + crop_img_size/2
        bottom = pos[1] + crop_img_size/2
        render = image.crop((left, up, right, bottom))
        mask_render = image.crop((left, up, right, bottom))
        offset = np.array([left, up])


    calib = np.load(os.path.join(folder, "PARAM" + suffix, subject, name_from_yaw(cam, yaw_pitch) + ".npy"), allow_pickle=True)
    Rt = calib.item().get("Rt")
    
    proj = calib.item().get('proj')
    mat = proj @ Rt
    clipping = np.array([[orig_img_size/2, 0, 0,orig_img_size/2],[0,-orig_img_size/2,0,orig_img_size/2],[0,0,1,0],[0,0,0,1]])
    mat = clipping @ mat
    mat = np.delete(mat, 2, 0)
    
    calibs.append(mat)
    

p = cv2.sfm.triangulatePoints(np.array(points), np.array(calibs))
# cv2::triangulatePoints() returns results in homo coo but not cv2::sfm::triangulatePoints()
# p = p / p[-1,:]
p = np.vstack((p, np.array([0])))
translation = p
np.save(os.path.join(folder, "PARAM", subject, "translation.npy"), p)


center = translation
center[3,0] = 1

points = []
calib = []
for cam in cams:
    render_filename = os.path.join(folder, "RENDER" + suffix, subject, name_from_yaw(cam, yaw_pitch) + ".jpg")
    if os.path.isfile(render_filename):
        image = Image.open(render_filename)
    else:
        image = Image.open(render_filename[:-4] + ".png")
    mask_image = Image.open(os.path.join(folder, "MASK" + suffix, subject, name_from_yaw(cam, yaw_pitch) + ".png"))
    
    mask = np.asarray(mask_image)
    idx = np.argwhere(mask == 255)
    nb_pix_height = idx[:,0].max(0) - idx[:,0].min(0)

    if nb_pix_height > 450:
        factor = np.random.randint(400,500) / nb_pix_height
        new_size = np.int(image.size[0] * factor)
        image = image.resize((new_size, new_size), Image.BILINEAR)
        mask_image = mask_image.resize((new_size, new_size), Image.NEAREST)
    else:
        factor = 1

    calib = np.load(os.path.join(folder, "PARAM" + suffix, subject, name_from_yaw(cam, yaw_pitch) + ".npy"), allow_pickle=True)
    Rt = calib.item().get("Rt")

    proj = calib.item().get('proj')
    mat = proj @ Rt
    clipping = np.array([[orig_img_size/2, 0, 0,orig_img_size/2],[0,-orig_img_size/2,0,orig_img_size/2],[0,0,1,0],[0,0,0,1]])
    mat = clipping @ mat
    mat = np.delete(mat, 2, 0)
    mat[0] *= factor
    mat[1] *= factor

    proj_center = mat @ center
    proj_center[0] /= proj_center[-1]
    proj_center[1] /= proj_center[-1]
    proj_center[2] /= proj_center[-1]
    
    size = 256
    p1 = proj_center[:2] - np.array([size, size]).reshape(2,1)
    p2 = proj_center[:2] + np.array([size, size]).reshape(2,1)
    p1 = np.floor(p1)
    p2 = np.floor(p2)

    render = image.crop((p1[0][0], p1[1][0], p2[0][0], p2[1][0]))
    mask_render = mask_image.crop((p1[0][0], p1[1][0], p2[0][0], p2[1][0]))
    
    offset = p1

    data = convert_calib_to_dict(calib)
    data["offset"] = p1
    data["resize"] = factor
    data["translation"] = translation
    np.save(os.path.join(folder, "INFERENCE/PARAM", subject, name_from_yaw(cam, yaw_pitch) + ".npy"), data)
    render.save(os.path.join(folder, "INFERENCE/RENDER", subject, name_from_yaw(cam, yaw_pitch) + ".jpg"), "JPEG")
    mask_render.save(os.path.join(folder, "INFERENCE/MASK", subject, name_from_yaw(cam, yaw_pitch) + ".png"), "PNG")