"""
Rendering with camera moving around the scene.
Object is placed randomly in the scene
Ortho and perspective
"""
from renderer.camera import Camera
import numpy as np
from renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl, save_obj_mesh
from renderer.camera import Camera
import os
import cv2
import time
import math
import random
import pyexr
import argparse
from tqdm import tqdm
import os.path
from os import path
import json

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def rotateSH(SH, R):
    SHn = SH
    
    # 1st order
    SHn[1] = R[1,1]*SH[1] - R[1,2]*SH[2] + R[1,0]*SH[3]
    SHn[2] = -R[2,1]*SH[1] + R[2,2]*SH[2] - R[2,0]*SH[3]
    SHn[3] = R[0,1]*SH[1] - R[0,2]*SH[2] + R[0,0]*SH[3]

    # 2nd order
    SHn[4:,0] = rotateBand2(SH[4:,0],R)
    SHn[4:,1] = rotateBand2(SH[4:,1],R)
    SHn[4:,2] = rotateBand2(SH[4:,2],R)

    return SHn

def rotateBand2(x, R):
    s_c3 = 0.94617469575
    s_c4 = -0.31539156525
    s_c5 = 0.54627421529

    s_c_scale = 1.0/0.91529123286551084
    s_c_scale_inv = 0.91529123286551084

    s_rc2 = 1.5853309190550713*s_c_scale
    s_c4_div_c3 = s_c4/s_c3
    s_c4_div_c3_x2 = (s_c4/s_c3)*2.0

    s_scale_dst2 = s_c3 * s_c_scale_inv
    s_scale_dst4 = s_c5 * s_c_scale_inv

    sh0 =  x[3] + x[4] + x[4] - x[1]
    sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4]
    sh2 =  x[0]
    sh3 = -x[3]
    sh4 = -x[1]
    
    r2x = R[0][0] + R[0][1]
    r2y = R[1][0] + R[1][1]
    r2z = R[2][0] + R[2][1]
    
    r3x = R[0][0] + R[0][2]
    r3y = R[1][0] + R[1][2]
    r3z = R[2][0] + R[2][2]
    
    r4x = R[0][1] + R[0][2]
    r4y = R[1][1] + R[1][2]
    r4z = R[2][1] + R[2][2]
    
    sh0_x = sh0 * R[0][0]
    sh0_y = sh0 * R[1][0]
    d0 = sh0_x * R[1][0]
    d1 = sh0_y * R[2][0]
    d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
    d3 = sh0_x * R[2][0]
    d4 = sh0_x * R[0][0] - sh0_y * R[1][0]
    
    sh1_x = sh1 * R[0][2]
    sh1_y = sh1 * R[1][2]
    d0 += sh1_x * R[1][2]
    d1 += sh1_y * R[2][2]
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
    d3 += sh1_x * R[2][2]
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2]
    
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y
    
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y
    
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    dst = x
    dst[0] = d0
    dst[1] = -d1
    dst[2] = d2 * s_scale_dst2
    dst[3] = -d3
    dst[4] = d4 * s_scale_dst4

    return dst

def render_prt(projection, out_path, folder_name, subject_name, shs, rndr, rndr_uv, width, height, angl_step=4, n_light=1, pitch=[0]):
    cam = Camera(width=width, height=height)
    
    if projection.startswith("ortho"):
        cam.ortho_ratio = 0.4
        cam.center = np.array([0, 0, 10])
        cam.near = -400
        cam.far = 400
    elif projection.startswith("persp"):
        cam.ortho_ratio = None
        cam.center = np.array([0, 0, 1200])
        cam.near = 10
        cam.far = 2000
        cam.focal_x = cam.focal_y = 3500
    else:
        print("Unknown projection")
        exit(0)

    cam.sanity_check()

    # set path for obj, prt
    # mesh_file = os.path.join(folder_name, subject_name + '.obj')
    mesh_file = os.path.join(folder_name, subject_name + '_100k.obj')
    if not os.path.exists(mesh_file):
        mesh_file = os.path.join(folder_name, subject_name + '_100k.OBJ')
    if not os.path.exists(mesh_file):
        print('ERROR: obj file does not exist!!', mesh_file)
        return 
    prt_file = os.path.join(folder_name, 'bounce', 'bounce0.txt')
    if not os.path.exists(prt_file):
        print('ERROR: prt file does not exist!!!', prt_file)
        return
    face_prt_file = os.path.join(folder_name, 'bounce', 'face.npy')
    if not os.path.exists(face_prt_file):
        print('ERROR: face prt file does not exist!!!', prt_file)
        return
    text_file = os.path.join(folder_name, 'tex', subject_name + '_dif_8k.jpg')
    if not os.path.exists(text_file):
        text_file = os.path.join(folder_name, 'tex', subject_name + '_dif.jpg')
    if not os.path.exists(text_file):
        print('ERROR: dif file does not exist!!', text_file)
        return             

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)

    """
    # Norm mesh height
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])
    vertices *= y_scale
    """

    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_med = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])

    

    motion_range = 200
    x_rand = (np.random.rand() * 2 - 1) * motion_range
    z_rand = (np.random.rand() * 2 - 1) * motion_range
    y_rand = 0
    pos = np.array([x_rand, y_rand - y_med, z_rand])
    vertices += pos

    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    prt = np.loadtxt(prt_file)
    face_prt = np.load(face_prt_file)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)    
    rndr.set_albedo(texture_image)

    rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)   
    rndr_uv.set_albedo(texture_image)

    os.makedirs(os.path.join(out_path, 'GEO', 'OBJ', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'PARAM', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'RENDER', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'MASK', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'KEYPOINTS', subject_name),exist_ok=True)

    os.makedirs(out_path,exist_ok=True)
    if not os.path.exists(os.path.join(out_path, 'val.txt')):
        f = open(os.path.join(out_path, 'val.txt'), 'w')
        f.close()

    # copy obj file
    mesh_out_file = mesh_file.split('/')[-1].replace("OBJ", "obj")
    cmd = 'cp %s %s' % (mesh_file, os.path.join(out_path, 'GEO', 'OBJ', subject_name, mesh_out_file))
    out_mesh = os.path.join(out_path, 'GEO', 'OBJ', subject_name, subject_name + '_100k.obj')
    save_obj_mesh(out_mesh, vertices, faces)
    # print(cmd)
    # os.system(cmd)
    
    
    # Save initial position, direction, up, and right
    cc = cam.center
    cd = cam.direction
    cr = cam.right
    cu = cam.up
    
    # original cam focal
    orig_focal_x = cam.focal_x
    orig_focal_y = cam.focal_y

    angl_step = 1

    yaw_pitch = {}

    for p in pitch:
        for y in tqdm(range(0, 360, angl_step)):

            p = np.random.randint(0,45)
            yaw_pitch[y] = p
            
            R = np.matmul(make_rotate(0, math.radians(y), 0), make_rotate(math.radians(-p), 0, 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90),0,0))

            cam.center = R @ cc
            cam.direction = R @ cd
            cam.right = R @ cr
            cam.up = R @ cu
            
            R = np.eye(3)

            rndr.rot_matrix = R
            rndr_uv.rot_matrix = R
            rndr.set_camera(cam)
            rndr_uv.set_camera(cam)

            for j in range(n_light):
                sh_id = random.randint(0,shs.shape[0]-1)

                # Remove random
                sh_id = 0

                sh = shs[sh_id]
                sh_angle = 0.2*np.pi*(random.random()-0.5)
                
                # Remove random
                sh_angle = 0

                sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)
                
                proj = rndr.projection_matrix
                norm = rndr.normalize_matrix
                view = rndr.model_view_matrix
                rot = rndr.rot_matrix
                
                if cam.ortho_ratio is not None:
                    K = np.identity(3)
                    K[0,0] =  1 / (0.5 * width * cam.ortho_ratio)
                    K[1,1] =  -1 / (0.5 * height * cam.ortho_ratio)
                else:
                    K = None
                
                dic = {"pos": pos, "rot" : rot, "K":K, "Rt":view, 'norm':norm, 'proj': proj,'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}

                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False
                rndr.display()

                out_all_f = rndr.get_color(0)
                out_mask = out_all_f[:,:,3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)

                np.save(os.path.join(out_path, 'PARAM', subject_name, '%d_%d_%02d.npy'%(y,p,j)),dic)
                cv2.imwrite(os.path.join(out_path, 'RENDER', subject_name, '%d_%d_%02d.jpg'%(y,p,j)),255.0*out_all_f)
                cv2.imwrite(os.path.join(out_path, 'MASK', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_mask)

    json_object = json.dumps(yaw_pitch, indent = 4)
    with open(os.path.join(out_path, 'RENDER', subject_name, "yaw_pitch.json"), "w") as f:
        f.write(json_object)


if __name__ == '__main__':
    shs = np.load('./render/env_sh.npy')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-m', '--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int, default=512, help='rendering image size')
    parser.add_argument('-p', '--projection', type=str, default='persp')
    args = parser.parse_args()

    # NOTE: GL context has to be created before any other OpenGL function loads.
    from renderer.gl.init_gl import initialize_GL_context
    initialize_GL_context(width=args.size, height=args.size, egl=args.egl)

    from renderer.gl.prt_render import PRTRender
    rndr = PRTRender(width=args.size, height=args.size, ms_rate=args.ms_rate, egl=args.egl)
    rndr_uv = PRTRender(width=args.size, height=args.size, uv_mode=True, egl=args.egl)

    if args.input[-1] == '/':
        args.input = args.input[:-1]
    subject_name = args.input.split('/')[-1]
    
    if subject_name.endswith("_OBJ_FBX"):
        subject_name = subject_name[:-8]
    elif subject_name.endswith("_OBJ"):
        subject_name = subject_name[:-4]

    # Change so that it doesn't process if the rendered image are already there
    out_folder = args.out_dir + "/RENDER/" + subject_name
    if path.exists(out_folder):
        print(out_folder, "alreay exists, Skip")
    else:
        render_prt(args.projection, args.out_dir, args.input, subject_name, shs, rndr, rndr_uv, args.size, args.size, 1, 1, pitch=[0])