import os
import numpy as np
import shutil
import json

def name_from_yaw(yaw, yaw_pitch):
    if yaw_pitch is None:
        return str(yaw)
    return str(yaw) + "_" + str(yaw_pitch[str(yaw)]) + "_00"

# PARAMETERS
# ==============================================
out_folder = ""
folder = ""
cameras = [0, 90, 180, 270]
subjects = [
    # "rp_{subject-name}_posed_{subject-id}"
]
suffix = "INFERENCE"
# ==============================================
if out_folder == "" or folder == "" or len(cameras) == 0 or len(subjects) == 0:
    print("Parameters should be specified")
    exit(0)

os.makedirs(out_folder, exist_ok=True)
for subject in subjects:
    with open(os.path.join(folder, "RENDER", subject, "yaw_pitch.json"), 'r') as f:
        yaw_pitch = json.load(f)
    shutil.rmtree(os.path.join(out_folder, subject), ignore_errors=True)

    os.makedirs(os.path.join(out_folder, subject), exist_ok=True)
    for cam in cameras:
        img_path = os.path.join(folder, suffix, "RENDER", subject, name_from_yaw(cam, yaw_pitch) + ".jpg")
        mask_path = os.path.join(folder, suffix, "MASK", subject, name_from_yaw(cam, yaw_pitch) + ".png")
        calib_path = os.path.join(folder, suffix, "PARAM", subject, name_from_yaw(cam, yaw_pitch) + ".npy")
        
        if os.path.isfile(img_path) == False:
            print(img_path)
            print("Image", cam, "doesn't exist, no humans found")
            # exit(0)
            continue
        
        shutil.copyfile(img_path, os.path.join(out_folder, subject, os.path.basename(img_path)))
        shutil.copyfile(mask_path, os.path.join(out_folder, subject, os.path.basename(mask_path)[:-4] + "_mask.png"))
        shutil.copyfile(calib_path, os.path.join(out_folder, subject, os.path.basename(calib_path)))