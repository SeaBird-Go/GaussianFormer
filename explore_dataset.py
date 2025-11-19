'''
Copyright (c) 2025 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2025-05-14 15:42:17
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import mmengine


def explore_pkl():
    # pkl_fp = "data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl"
    pkl_fp = "data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl"
    data = mmengine.load(pkl_fp)
    scene_infos = data['infos']  # len = number of scenes
    keyframes = data['metadata']

    print(len(scene_infos))
    print(len(scene_infos['e7ef871f77f44331aefdebc24ec034b7']))
    # print(scene_infos['e7ef871f77f44331aefdebc24ec034b7'][1])

    for i, keyframe in enumerate(keyframes):
        x = keyframe
        print(keyframe)
        _info = x[0] + "{:0>3}".format(str(x[1]))
        print(_info)
        if i>10:
            break

    keyframes = sorted(keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
    print(len(keyframes))

    idx = 997
    scene_token, index = keyframes[idx]
    info = scene_infos[scene_token][index]
    print(info)


def concat_nuscenes_images(cam_imgs, spacing=5):
    """Concat the 6 nuscenes camera image into one image.
    Args:
        cam_imgs (list[PIL.Image]): List of 6 camera images.
        spacing (int, optional): Spacing between images. Defaults to 5.
    Returns:
        PIL.Image: The concatenated image.
    """
    assert len(cam_imgs) == 6, "There should be 6 camera images to concatenate."

    cam_img_size = cam_imgs[0].size
    cam_w, cam_h = cam_img_size
    result_w = cam_w * 3 + 2 * spacing
    result_h = cam_h * 2 + 1 * spacing
    result = Image.new(cam_imgs[0].mode, (result_w, result_h), (0, 0, 0))

    result.paste(cam_imgs[0], box=(1*cam_w+1*spacing, 0))
    result.paste(cam_imgs[1], box=(2*cam_w+2*spacing, 0))
    result.paste(cam_imgs[2], box=(0, 0))
    result.paste(cam_imgs[3], box=(1*cam_w+1*spacing, 1*cam_h+1*spacing))
    result.paste(cam_imgs[5], box=(0, 1*cam_h+1*spacing))
    result.paste(cam_imgs[4], box=(2*cam_w+2*spacing, 1*cam_h+1*spacing))
    return result

def save_all_cam_cat_images(anno_file, save_root=None):
    """Concatenate all the camera images into one image.

    Args:
        anno_file (_type_): _description_
        save_root (_type_, optional): _description_. Defaults to None.
    """
    dataset = mmengine.load(anno_file)
    
    scene_infos = dataset['infos']
    keyframes = dataset['metadata']
    keyframes = sorted(keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

    sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    data_root = "./data/nuscenes"
    cam_img_size = [480, 270]  # [w, h]

    save_root = "./data/nuscenes_val_concat"
    for idx, _info in tqdm(enumerate(keyframes), total=len(keyframes)):
        scene_token, index = _info
        info = scene_infos[scene_token][index]

        cam_imgs = []
        for cam_name in sensor_types:
            cam_data = info['data'][cam_name]
            filename = osp.join(data_root, cam_data['filename'])

            cam_img_resized = Image.open(filename).resize(
                cam_img_size, Image.BILINEAR)
            cam_imgs.append(cam_img_resized)
        
        result = concat_nuscenes_images(cam_imgs, spacing=0)

        if save_root is not None:
            # create the camera image directory
            os.makedirs(save_root, exist_ok=True)

            cam_img_path = osp.join(save_root, f"{idx:06d}.jpg")
            result.save(cam_img_path)


def save_video():
    # list the data dir
    data_dir = "data/nuscenes_val_concat"
    img_list = sorted(os.listdir(data_dir))
    img_list = [osp.join(data_dir, img) for img in img_list]

    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10

    video_dir = "data/nuscenes_val_concat_video"
    os.makedirs(video_dir, exist_ok=True)

    video = cv2.VideoWriter(osp.join(video_dir, 'output.mp4'), fourcc, fps, (1440, 540))

    for idx, _fp in tqdm(enumerate(img_list)):
        frame = cv2.imread(_fp)
        # put a text to represent the frame id on the image
        cv2.putText(frame, f"Frame {idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        video.write(frame)

    video.release()
    print('Video saved as output.mp4')


def save_frame_idx_to_scene_info(anno_file, save_root=None, split="val"):
    """Save the frame index to scene name and scene token offline.

    Args:
        anno_file (_type_): _description_
        save_root (_type_, optional): _description_. Defaults to None.
    """
    dataset = mmengine.load(anno_file)
    
    scene_infos = dataset['infos']
    keyframes = dataset['metadata']
    keyframes = sorted(keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

    save_root = "./data"
    output_str_list = []
    for idx, _info in tqdm(enumerate(keyframes), total=len(keyframes)):
        scene_token, index = _info
        info = scene_infos[scene_token][index]

        sample_token = info['token']
        scene_token = info['scene_token']

        output_str = f"{idx} {sample_token} {scene_token}"
        output_str_list.append(output_str)
    
    head = "frame_idx sample_token scene_token\n"
    if save_root is not None:
        output_str = "\n".join(output_str_list)
        
        save_file = osp.join(save_root, f"frame_idx_2_secene_info_{split}.txt")
        with open(save_file, "w") as fp:
            fp.write(output_str)


if __name__ == "__main__":
    # explore_pkl()

    pkl_fp = "data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl"
    
    # save_all_cam_cat_images(pkl_fp)

    save_frame_idx_to_scene_info(pkl_fp)

    # save_video()