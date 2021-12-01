import os
from shutil import copyfile

# PARAMETERS
# ============================================
base_path = ""
output_folder = ""
angles = [0, 90, 180, 270]
data = {
    # "rp_{subject-name}_posed_{subject-id}": angles,
}
pitchs = [0] * len(angles)
copy_mesh = False
# ============================================
if base_path == "" or output_folder == "" or len(angles) == 0 or len(data) == 0:
    print("Parameters should be specified")
    exit(0)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_folder = "RENDER"
mask_folder = "MASK"
calib_folder = "PARAM"

for subject in data:
    
    if len(angles) == 1:
        folder = output_folder + "/"
        for view in data[subject]:
            copyfile(base_path + image_folder + "/" + subject + "/" + str(view) + "_0_00.jpg", folder + subject + ".jpg")
            copyfile(base_path + mask_folder + "/" + subject + "/" + str(view) + "_0_00.png", folder + subject + "_mask.png")

    else:
        folder = output_folder + subject + "/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i, view in enumerate(data[subject]):
            suffix = "_00"
            view_pitch = str(view) + "_" + str(pitchs[i])
            copyfile(base_path + image_folder + "/" + subject + "/" + view_pitch + suffix + ".jpg", folder + view_pitch + suffix + ".jpg")
            copyfile(base_path + mask_folder + "/" + subject + "/" + view_pitch + suffix + ".png", folder + view_pitch + suffix + "_mask.png")
            copyfile(base_path + calib_folder + "/" + subject + "/" + view_pitch + suffix + ".npy", folder + view_pitch + suffix + ".npy")
            if copy_mesh and not os.path.isfile(folder + subject + "_100k.obj") and os.path.isdir(base_path + "GEO/OBJ/"):
                copyfile(base_path + "GEO/OBJ/" + subject + "/" + subject + "_100k.obj", folder + subject + "_100k.obj")