import pyproj
import simplekml
import numpy as np
import cv2
import math

# extended libraries for extracting GPS ground truth from drone taken images
import re
import os
import simplekml


def update_current_GPS(sat_gps, pix_c):
    GSD = [0.1493, -0.1492] # m/pix
    # convert initial GPS to projective distance in meters
    # geo_epsg, proj_epsg = "epsg:4326", "epsg:3857"
    transformer = pyproj.Transformer.from_crs(4326, 3857)
    init_proj = transformer.transform(*sat_gps)
    current_proj = [init_proj[i]+pix_c[i]*GSD[i] for i in range(len(init_proj))]
    
    # convert current projective distance to GPS
    transformer = pyproj.Transformer.from_crs(3857, 4326)
    current_GPS = transformer.transform(*current_proj)
    return current_GPS


def retrieve_init_pixposition(sat_gps, init_gps):
    GSD = [0.1493, -0.1492] # m/pix
    # convert initial GPS to projective distance in meters
    # geo_epsg, proj_epsg = "epsg:4326", "epsg:3857"
    transformer = pyproj.Transformer.from_crs(4326, 3857)
    sat_proj = transformer.transform(*sat_gps)
    init_proj = transformer.transform(*init_gps)
    
    pixpos = [int((init_proj[i]-sat_proj[i])/GSD[i]) for i in range(len(init_proj))]
    return pixpos


def generate_kml(GPS, is_gt=True):
    kml=simplekml.Kml()

    start_pt = kml.newpoint(name='Start Point')
    start_pt.coords = [GPS[0]]
    start_pt.style.labelstyle.scale = 1  # Make the text twice as big
    
    start_pt.style.labelstyle.color = simplekml.Color.white
    start_pt.altitudemode = simplekml.AltitudeMode.relativetoground

    end_pt = kml.newpoint(name='End Point')
    end_pt.coords = [GPS[-1]]
    end_pt.style.labelstyle.scale = 1  # Make the text twice as big
    
    end_pt.style.labelstyle.color = simplekml.Color.white
    end_pt.altitudemode = simplekml.AltitudeMode.relativetoground

    ls = kml.newlinestring(name='3D Path', extrude=1)
    ls.coords = GPS
    ls.extrude = 1
    ls.style.linestyle.width = 3
    if is_gt:
        ls.style.linestyle.color = simplekml.Color.red
        end_pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-blank.png'
        start_pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-blank.png'
    else:
        ls.style.linestyle.color = simplekml.Color.yellow
        start_pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/grn-blank.png'
        end_pt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/grn-blank.png'
    ls.altitudemode = simplekml.AltitudeMode.relativetoground
    return kml

def UAV_loc_by_pix_DLT(image1, mkpts0, mkpts1, UAV_pix_pos_offset, opt, bins=0):
    size = opt.resize
    H1, W1 = image1.shape
    angle = bins*opt.bin_interval
    
    # project image1 boundaries to image0
    src_pts = np.float32(mkpts1).reshape(-1,1,2)
    dst_pts = np.float32(mkpts0).reshape(-1,1,2)
    
    hom_reproj_threshold = 3.0  # threshold for homography reprojection error: maximum allowed reprojection error in pixels (to treat a point pair as an inlier)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=hom_reproj_threshold)
    
    abs_sin = abs(math.sin(math.radians(angle)))
    abs_cos = abs(math.cos(math.radians(angle)))
    img_box = np.float32([[size[1]*abs_sin-1,H1],
                          [0,size[0]*abs_sin-1],
                          [size[0]*abs_cos-1,0],
                          [W1,size[1]*abs_cos-1]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(img_box, M)
    points= np.int32(dst)
    M = cv2.moments(points)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = (cx, cy)
    
    return center, points, mask


def UAV_loc_by_pix_PAffine(image1, mkpts, UAV_pix_pos_offset, opt, vs_scale, not_updated, bins=0):
    # partial affine (rotation, scale and translation)
    use_ground, mkpts0, mkpts1, mkpts0_other, mkpts1_other = mkpts
    size = opt.resize
    H1, W1 = image1.shape
    angle = bins*opt.bin_interval
    scale_ground, scale_building = None, None
    
    # project image1 boundaries to image0
    src_pts = np.float32(mkpts1).reshape(-1,2)
    dst_pts = np.float32(mkpts0).reshape(-1,2)
    
    reproj_threshold = 3.0  # threshold for homography reprojection error: maximum allowed reprojection error in pixels (to treat a point pair as an inlier)
    Mtx, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)

    theta = np.degrees(np.arctan(Mtx[1,0]/Mtx[0,0]))
    offset = round(theta/opt.bin_interval)
    if use_ground:
        scale_ground = Mtx[0,0]/np.cos(np.radians(theta))
        # compute building scale
        if len(mkpts0_other)>opt.switch_threshold:
            src_pts = np.float32(mkpts1_other).reshape(-1,2)
            dst_pts = np.float32(mkpts0_other).reshape(-1,2)
            Mtx_scale, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
            scale_building = Mtx_scale[0,0]/np.cos(np.radians(np.degrees(np.arctan(Mtx_scale[1,0]/Mtx_scale[0,0]))))
    else:
        scale_building = Mtx[0,0]/np.cos(np.radians(theta))
        # compute ground scale
        if len(mkpts0_other)>opt.switch_threshold:
            src_pts = np.float32(mkpts1_other).reshape(-1,2)
            dst_pts = np.float32(mkpts0_other).reshape(-1,2)
            Mtx_scale, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
            scale_ground = Mtx_scale[0,0]/np.cos(np.radians(np.degrees(np.arctan(Mtx_scale[1,0]/Mtx_scale[0,0]))))
    avg_building_h = 111.67527558*abs(scale_ground-scale_building) if (scale_ground and scale_building) is not None else np.nan
    # print(scale_ground, scale_building)
    upper_bound, lower_bound = 1.035816**((not_updated+1)*opt.KF_dt), 0.964184**((not_updated+1)*opt.KF_dt)
    scale_ground = 1.0 if scale_ground is None else max(min(scale_ground,upper_bound),lower_bound)

    # retrieve resize image four vertex
    threshold = 360/opt.bin_interval
    img_box = retrieve_img_box(H1, W1, size, vs_scale, angle, bins, threshold)
    
    dst = img_box@Mtx.T # n*3@3*2
    points= np.int32(dst)
    M = cv2.moments(points)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = (cx, cy)

    return center, points, img_box, mask, offset, scale_ground, avg_building_h


def UAV_loc_by_pix_Affine(image1, mkpts0, mkpts1, UAV_pix_pos_offset, opt, bins=0):
    # partial affine(above) + shearring
    size = opt.resize
    H1, W1 = image1.shape
    angle = bins*opt.bin_interval
    
    # project image1 boundaries to image0
    src_pts = np.float32(mkpts1).reshape(-1,2)
    dst_pts = np.float32(mkpts0).reshape(-1,2)
    
    reproj_threshold = 3.0  # threshold for homography reprojection error: maximum allowed reprojection error in pixels (to treat a point pair as an inlier)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reproj_threshold)
    
    threshold = 360/opt.bin_interval
    img_box = retrieve_img_box(H1, W1, size, angle, bins, threshold)

    dst = img_box@M.T # n*3@3*2

    points= np.int32(dst)
    M = cv2.moments(points)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = (cx, cy)
    
    current_GPS = update_current_GPS(opt.satmap_init_gps, (cx+UAV_pix_pos_offset[0], cy+UAV_pix_pos_offset[1]))
    return current_GPS, center, points, mask


def retrieve_img_box(H1, W1, size, vs_scale, angle, bins, threshold):
    abs_sin = abs(math.sin(math.radians(angle)))
    abs_cos = abs(math.cos(math.radians(angle)))
    size_w, size_h = int(size[0]*vs_scale), int(size[1]*vs_scale)
    
    if 0<=angle<=90 or 180<=angle<=270: # bins 45-60, 15-30
        img_box = np.float32([[size_w*abs_cos,0,1],
                              [0,size_w*abs_sin,1],
                              [size_h*abs_sin,H1,1],
                              [W1,size_h*abs_cos,1]]).reshape(-1,3)
    else: # bins 0-15, 30-45
        img_box = np.float32([[size_h*abs_sin,0,1],
                              [0,size_h*abs_cos,1],
                              [size_w*abs_cos,H1,1],
                              [W1,size_w*abs_sin,1]]).reshape(-1,3)
    return img_box