from pathlib import Path
import argparse
import numpy as np
import torch
import json
import os

from models.matching import Matching
from models.utils.utils import (AverageTimer, VideoStreamer, load_encoder_img, frame2tensor)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1280, 720],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--map_row_col', type=int, nargs='+', default=[4,4],
        help='Map composed with row*col sub-maps')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    
    kpts = {'keypoints0':np.empty([0,2]),
        'scores0':np.empty([0]),
        'descriptors0':np.empty([256,0]),
        'image0':np.empty([opt.resize[1]*opt.map_row_col[0], opt.resize[0]*opt.map_row_col[1]])}

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Load timer and dataloader
    print('==> Processing image directory input: {}'.format(opt.input))
    img_dirs = []
    for i in range(opt.map_row_col[0]):
        for j in range(opt.map_row_col[1]):
            dir = 'sat_{}_{}.png'.format(i,j)
            img_dirs.append(opt.input+dir)

    for i, imdir in enumerate(img_dirs):
        frame = load_encoder_img(imdir, opt.resize)
        frame_tensor = frame2tensor(frame, device)
        last_data = matching.superpoint({'image': frame_tensor})
        last_data = {k+'0': last_data[k][0].cpu().numpy() for k in keys}

        row = opt.resize[1]*(i//opt.map_row_col[1])
        col = opt.resize[0]*(i%opt.map_row_col[1])
        print('row,col:', row, col)
        
        # Reorgnize keypoints
        last_data['keypoints0'] = last_data['keypoints0']+np.array([col,row])
        kpts['keypoints0'] = np.concatenate((kpts['keypoints0'],last_data['keypoints0']), axis=0)
        
        kpts['scores0'] = np.concatenate((kpts['scores0'],last_data['scores0']), axis=0)
        kpts['descriptors0'] = np.concatenate((kpts['descriptors0'],last_data['descriptors0']), axis=1)
        kpts['image0'][row:row+opt.resize[1], col:col+opt.resize[0]] = frame

    image0_info = {'keypoints0':kpts['keypoints0'],
                    'scores0':kpts['scores0'],
                    'descriptors0':kpts['descriptors0'],
                    'image0':kpts['image0']}
    
    # save kpts into npz file
    np.savez(opt.output_dir+'/satmap_kpts.npz', **image0_info)
