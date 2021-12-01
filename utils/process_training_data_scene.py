import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import trimesh
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import shutil
import argparse


def convert_calib_to_dict(calib):
    res = {}
    for i in ["pos", "rot", "K", "Rt", 'norm', 'proj', 'sh', 'ortho_ratio', 'scale', 'center', 'R']:
        res[i] = calib.item().get(i)
    return res

def crop_subject(subject):
    print(subject)
    # Skip if existing
    if os.path.isdir(os.path.join(folder, "PARAM", subject)):
        path, dirs, files = next(os.walk(os.path.join(folder, "PARAM", subject)))
        file_count = len(files)
        if file_count > 0:
            print(subject, ": Already exists, Skip")
            return 0

    # print(subject)
    os.makedirs(os.path.join(folder, "RENDER", subject), exist_ok=True)
    os.makedirs(os.path.join(folder, "MASK", subject), exist_ok=True)
    os.makedirs(os.path.join(folder, "PARAM", subject), exist_ok=True)

    with open(os.path.join(folder, "RENDER_orig", subject, "yaw_pitch.json"), 'r') as f:
        yaw_pitch = json.load(f)
    json_object = json.dumps(yaw_pitch, indent = 4)
    with open(os.path.join(folder, 'RENDER', subject, "yaw_pitch.json"), "w") as f:
        f.write(json_object)

    # for cam in tqdm(range(num_cams)):
    for cam in range(num_cams):
        calib = np.load(os.path.join(folder, "PARAM_orig", subject, str(cam) + "_" + str(yaw_pitch[str(cam)]) + "_00.npy"), allow_pickle=True)
        proj = calib.item().get('proj')
        view = calib.item().get('Rt')
        pos = calib.item().get("pos").reshape(3,1)
        center = calib.item().get("center").reshape(3,1)
        pos += center
        pos = np.vstack((pos, 1))
        P = proj @ view
        
        image = Image.open(os.path.join(folder, "RENDER_orig", subject, str(cam) + "_" + str(yaw_pitch[str(cam)]) + "_00.jpg"))
        mask_image = Image.open(os.path.join(folder, "MASK_orig", subject, str(cam) + "_" + str(yaw_pitch[str(cam)]) + "_00.png"))
        mask = np.asarray(mask_image)
        idx = np.argwhere(mask == 255)
        nb_pix_height = idx[:,0].max(0) - idx[:,0].min(0)

        
        if nb_pix_height > 500:
            factor = np.random.randint(450,500) / nb_pix_height
            print("Resize: ", subject, cam, factor)
            new_size = np.int(image.size[0] * factor)
            image = image.resize((new_size, new_size), Image.BILINEAR)
            mask_image = mask_image.resize((new_size, new_size), Image.NEAREST)
        else:
            factor = 1


        clipping = np.array([
                                [orig_img_size/2, 0, 0, orig_img_size/2],
                                [0, -orig_img_size/2, 0, orig_img_size/2],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]
                            ])
        clipping[0] *= factor
        clipping[1] *= factor
        view_vertices = clipping @ P @ pos
        view_vertices[0] /= view_vertices[3]
        view_vertices[1] /= view_vertices[3]
        view_vertices[2] /= view_vertices[3]
        projection = view_vertices


        if debug:
            fig = plt.figure()
            plt.scatter(projection[0, :], projection[1, :], s=1, marker='o', c="green")
            plt.imshow(image)
            plt.show()


        size = 256
        p1 = projection[:2] - np.array([size, size]).reshape(2,1)
        p2 = projection[:2] + np.array([size, size]).reshape(2,1)
            
        p1 = np.floor(p1)
        p2 = np.floor(p2)

        render = image.crop((p1[0][0], p1[1][0], p2[0][0], p2[1][0]))
        mask_render = mask_image.crop((p1[0][0], p1[1][0], p2[0][0], p2[1][0]))
        
        render = render.resize((crop_img_size, crop_img_size), Image.BILINEAR)
        mask_render = mask_render.resize((crop_img_size, crop_img_size), Image.NEAREST)

        data = convert_calib_to_dict(calib)
        data["offset"] = p1
        data["resize"] = factor


        if debug:
            mesh_path = os.path.join(folder, "GEO/OBJ", subject, subject+  "_100k.obj")
            mesh = trimesh.load(mesh_path)
            idx = np.random.choice(mesh.vertices.shape[0], 1000, replace=False)
            points = mesh.vertices[idx]
            points = np.concatenate((points.T, np.ones((1, points.shape[0]))))

            clipping[0] *= factor
            clipping[1] *= factor
            view_vertices = clipping @ P @ points
            view_vertices[0] /= view_vertices[3]
            view_vertices[1] /= view_vertices[3]
            view_vertices[2] /= view_vertices[3]
            view_vertices = view_vertices[:2,:]
            view_vertices -= p1


            fig = plt.figure()
            plt.scatter(view_vertices[0, :], view_vertices[1, :], s=1, marker='o', c="green")
            plt.imshow(render)
            plt.show()
            exit(0)

        np.save(os.path.join(folder, "PARAM", subject, str(cam) + "_" + str(yaw_pitch[str(cam)]) + "_00.npy"), data)
        render.save(os.path.join(folder, "RENDER", subject, str(cam) + "_" + str(yaw_pitch[str(cam)]) + "_00.jpg"), "JPEG")
        mask_render.save(os.path.join(folder, "MASK", subject, str(cam) + "_" + str(yaw_pitch[str(cam)]) + "_00.png"), "PNG")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-n', '--num_process', type=int, default=mp.cpu_count(), help='Number of process')
    args = parser.parse_args()
    folder = args.input
    num_cpu = args.num_process

    crop_img_size = 512
    orig_img_size = 2048
    debug = False
    num_cams = 360

    if not os.path.isdir(os.path.join(folder, "RENDER_orig")):
        render_folder = os.path.join(folder, "RENDER")
    else:
        render_folder = os.path.join(folder, "RENDER_orig")
    subjects = [os.path.basename(os.path.join(render_folder, o)) for o in os.listdir(render_folder) if os.path.isdir(os.path.join(render_folder,o))]



    if not (os.path.isdir(os.path.join(folder, "PARAM_orig")) or os.path.isdir(os.path.join(folder, "RENDER_orig")) or os.path.isdir(os.path.join(folder, "MASK_orig"))):
        print("Moved folders xxx into xxx_orig")
        shutil.move(os.path.join(folder, "PARAM"), os.path.join(folder, "PARAM_orig"))
        shutil.move(os.path.join(folder, "RENDER"), os.path.join(folder, "RENDER_orig"))
        shutil.move(os.path.join(folder, "MASK"), os.path.join(folder, "MASK_orig"))
    pool = Pool(num_cpu)
    results = pool.map(crop_subject, subjects)
