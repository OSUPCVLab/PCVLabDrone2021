from pathlib import Path
import os
import gc
import argparse
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import numpy as np
import matplotlib.cm as cm
from pyqtree import Index
import pickle
import torch
import time

from models.matching import Matching
from models.utils.utils import AverageTimer, VideoStreamer, frame2tensor, remove_kpts_on_building, segment_keypoints, update_last_data
from models.utils.utils_loc import generate_kml, retrieve_init_pixposition, update_current_GPS, UAV_loc_by_pix_PAffine
from models.utils.utils_plot import make_localization_plot

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='./assets/DJI_data/images/',
        help='URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default='./output/images/',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1280, 720],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--apply_GIS', action='store_true',
        help='segment matches keypoints from building and non-building')
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
        '--switch_threshold', type=int, default=50,
        help='Threshold for switching keypoints from non-building to building')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Patience for early stopping if UAV position was not updated over 10 seconds (video) or 10 frames(images), 0 is off.')
    parser.add_argument(
        '--KF_dt', type=float, default=1.0,
        help='Time between steps in seconds')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--matching_vis', action='store_true',
        help='Show the matched pairs')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--satmap_init_gps', type=float, nargs='+', default=[40.01872927, -83.033835], # large sat
        help='GPS of top-left corner of satellite map')
    parser.add_argument(
        '--Init_GPS', type=float, nargs='+', default=[40.012701, -83.009691], # Demo starting point GPS
        help='Initial drone flight GPS')
    parser.add_argument(
        '--Orien', type=float, default=0.0,
        help='UAV initial orientation is the angel to initially rotate first image clockwise to North direction, ranging from 0-360.')
    parser.add_argument(
        '--Init_height', type=float, default=None,
        help='UAV initial flight height')
    parser.add_argument(
        '--bin_interval', type=int, default=10,
        help='Divide 360 degrees into multiple bins, each bin shares certain degrees')
    parser.add_argument(
        '--range', type=int, nargs='+', default=[900, 900],
        help='Crop partial satellite image size (WxH) as basemap for matching')
    parser.add_argument(
        '--update_freq', type=int, default=3,
        help='Basemap update frequency. Update basemap once UAV center moves out of 1/k basemap range')

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

    timer = AverageTimer()
    # Load sat map info and its quadtree indexing file
    satmap_kpts = np.load('./featurebase/satmap_kpts.npz')
    image0, keypoints0, descriptors0, scores0 = satmap_kpts['image0'], satmap_kpts['keypoints0'], satmap_kpts['descriptors0'], satmap_kpts['scores0']
    del satmap_kpts; gc.collect()
    print('Satellite image size is {}x{} (HxW), containing {} keypoints'.format(*image0.shape, len(keypoints0)))
    print('Max basemap range is {}x{} (WxH)'.format(*opt.range))
    timer.update('Successfully loaded satellite map data, loading time',printout=True)
    
    if os.path.exists('./featurebase/QuadTree_idx.pkl'):
        with open('./featurebase/QuadTree_idx.pkl', 'rb') as inp:
            spindex = pickle.load(inp)
    else:
        spindex = Index(bbox=(0, 0, image0.shape[1], image0.shape[0]))  # Area of WxH
        for i in range(len(keypoints0)):
            w, h = keypoints0[i]
            spindex.insert(i, (w,h,w,h))
        # save quadtree indexing
        with open('./featurebase/QuadTree_idx.pkl', 'wb') as outp:
            pickle.dump(spindex, outp, pickle.HIGHEST_PROTOCOL)
    timer.update('Successfully loaded satellite keypoints quadtree indexing, loading time',printout=True)

    # Load satellite image GIS labels
    mask = np.asarray(Image.open('./featurebase/GIS_mask.png'), dtype=np.int32) if opt.apply_GIS else None
    timer.update('Successfully loaded GIS data, loading time',printout=True)
    
    # Initialize frame0 (last_data) at the beginning
    c_w, c_h = retrieve_init_pixposition(opt.satmap_init_gps, opt.Init_GPS)# basemap center in pixel distance in reference to top-left corner of satellite map
    r_w, r_h = min(opt.range[0], c_w), min(opt.range[1], c_h) # in case it reaches satmap boundary
    xmin, ymin, xmax, ymax = c_w-r_w, c_h-r_h, c_w+r_w, c_h+r_h
    base_map = image0[ymin:ymax, xmin:xmax]
    UAV_pix_pos_offset = [c_w-r_w, c_h-r_h]
    
    timer.reset()
    last_data, labels = update_last_data((image0, keypoints0, descriptors0, scores0), mask, spindex, (xmin, ymin, xmax, ymax), device) # return updated GIS labels if required
    timer.update('Successfully updated last data, updating time',printout=True)

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # dataloader
    vs = VideoStreamer(opt)
    frame, ret = vs.next_frame(1.0, go_next=False)
    assert ret, 'Error when reading the first frame (try different --input?)'
    

    # Initial parameters setup
    timer = AverageTimer()
    center, height = (r_w, r_h), opt.Init_height
    not_valid, points, img_box = None, None, None
    GPS = [] # save GPS as kml file which could be visualized at Google Earth
    pred_GPS = opt.Init_GPS
    Bins = round(opt.Orien/opt.bin_interval)
    not_updated, offset, update_scale = 0, 0, 1.0

    while True:
        # update UAV rotation bins
        Bins -= offset
        Bins = (360/opt.bin_interval+Bins) if Bins<0 else Bins%(360/opt.bin_interval)
        
        # update basemap range if center shift over range/2
        if abs(center[0]-r_w)>r_w/opt.update_freq or abs(center[1]-r_h)>r_h/opt.update_freq:
            c_w, c_h = center[0]+UAV_pix_pos_offset[0], center[1]+UAV_pix_pos_offset[1]
            r_w, r_h = min(opt.range[0], c_w), min(opt.range[1], c_h) # in case it reaches satmap boundary
            xmin, ymin, xmax, ymax = c_w-r_w, c_h-r_h, c_w+r_w, c_h+r_h
            last_data, labels = update_last_data((image0, keypoints0, descriptors0, scores0), mask, spindex, (xmin, ymin, xmax, ymax), device) # return updated GIS labels if required
            base_map = image0[ymin:ymax, xmin:xmax]
            center, UAV_pix_pos_offset = (r_w, r_h), [c_w-r_w, c_h-r_h]
        
        frame, ret = vs.next_frame(update_scale, rotate=True, bins=Bins)
        if not ret or not_updated>opt.patience:
            print('Finished UAV Geolocalization Inference')
            break
        stem1 = vs.i-1
        timer.update('data')

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        
        valid = matches > -1
        if opt.apply_GIS:
            valid, not_valid, use_ground, mkpts_count = segment_keypoints(valid, labels, opt.switch_threshold)
        
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        # keep matched keypoints not selected
        mkpts0_other = kpts0[not_valid]
        mkpts1_other = kpts1[matches[not_valid]]

        color = cm.jet(confidence[valid])
        timer.update('Matching')

        # Geolocalize UAV once matches keypoints are over 50
        if len(mkpts0)>=opt.switch_threshold:
            mkpts = (use_ground, mkpts0, mkpts1, mkpts0_other, mkpts1_other)
            # Geolocalize UAV with matched keypoints
            center, points, img_box, M, offset, update_scale, avg_building_h = UAV_loc_by_pix_PAffine(frame, mkpts, UAV_pix_pos_offset, opt, vs.scale, not_updated, bins=Bins)
            current_GPS = update_current_GPS(opt.satmap_init_gps, (center[0]+UAV_pix_pos_offset[0], center[1]+UAV_pix_pos_offset[1]))
            height = -1.23904244+vs.scale*111.67527558
            GeoLoc, not_updated = True, 0
        else:
            GeoLoc, offset = False, 0 # Initialize rotation offset
            not_updated = not_updated+1 # Not able to geolocalize UAV, not_updated count+1
            M, update_scale = [], 1.0 # Zeroize PAffine transformation mask and scale if unable to geolocalize UAV
            print('Don\'t have enough matched keypoint pairs over {} frames'.format(not_updated))

        if GeoLoc:
            GPS.append([stem1, *current_GPS])
        timer.update('Geolocalization')
        
        # Visualize the matches.
        if opt.matching_vis:
            text = [
                'Estimated GPS: ({:.6f}, {:.6f})'.format(*current_GPS),
                'Heading Direction (degrees): {}'.format(int(360-Bins*opt.bin_interval)%360), # heading_direction  = 360 - rotation_angle_offset
                'Flight Height (meters): {}'.format(int(round(height)))
            ]
            
            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Ground/Building/Total: {}/{}/{}'.format(*mkpts_count, sum(mkpts_count)),
                'Inliers pct: {:.2f}%'.format(np.sum(M)/len(M)*100),
                'Scale/Update_scale : {:.2f}/{:.4f}'.format(vs.scale, update_scale)
            ]

            out = make_localization_plot(GeoLoc, base_map, frame, kpts0, kpts1, mkpts0, mkpts1,color, opt.resize, center, points,
                                         img_box, text, path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
            out = cv2.resize(out, (0,0), fx=1/2, fy=1/2)

            # save sat image and frame t matched output
            if opt.output_dir is not None:
                stem = 'matches_{:06}'.format(stem1)
                out_file = str(Path(opt.output_dir, stem + '.png'))
                print('\n\nWriting image to {}'.format(out_file))
                cv2.imwrite(out_file, out)
            timer.update('Matching Vis')
        
        timer.print(text='Timer {:04d}'.format(stem1))

    cv2.destroyAllWindows()
    vs.cleanup()
    
    # save predicted GPS to .txt file
    # save predicted current GPS
    f = open(opt.output_dir+"GPS_pred.txt", "w")
    for item in GPS:
        f.write(f'{item[0]}\t{item[1]}\t{item[2]}\n')
    f.close()
    
    # save predicted GPS as .kml file
    GPS_kml = [(item[2], item[1], 1.0) for item in GPS]
    kml = generate_kml(GPS_kml, is_gt=False)
    kml.save(str(Path(opt.output_dir, 'GPS_pred.kml')))
    print('Saving predicted UAV GPS as .txt and .kml file')
    print('Inference done!')
