from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import math
from vidgear.gears import CamGear
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default', printout=False):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now
        if printout:
            print('%s=%.2f s' %(name, dt))

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()

def load_encoder_img(impath, resize):
    """ Read image as grayscale and resize to img_size.
    Inputs
        impath: Path to input image.
    Returns
        grayim: uint8 numpy array sized H x W.
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    w, h = grayim.shape[1], grayim.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    grayim = cv2.resize(
        grayim, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return grayim


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, opt):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = opt.resize
        self.scale = opt.Init_height*0.00895404+0.01114674 if opt.Init_height else 1.0
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = opt.skip
        self.bin_interval = opt.bin_interval
        self.max_length = opt.max_length
        basedir = opt.input
        image_glob = opt.image_glob
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            # Available Streams are: [144p, 240p, 360p, 480p, 720p, 1080p, best, worst]
            options = {"STREAM_RESOLUTION": "720p", 'CAP_PROP_FPS':5, "THREADED_QUEUE_MODE": False}
            self.stream = CamGear(source=basedir, stream_mode = True, logging=True, **options).start() # YouTube Video URL as input
            self._ip_camera = True
            self.listing = range(0, self.max_length)
            opt.KF_dt = 1.0/options['CAP_PROP_FPS']
            opt.patience = int(opt.patience*options['CAP_PROP_FPS']/opt.skip)
            print('==> Stop if UAV GPS not updated over {} frames'.format(opt.patience))
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
            print('==> Stop if UAV GPS not updated over {} frames'.format(opt.patience))
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
            opt.KF_dt = 1.0/(self.cap.get(cv2.CAP_PROP_FPS)/opt.skip)
            opt.patience = int(opt.patience*self.cap.get(cv2.CAP_PROP_FPS)/opt.skip)
            print('==> Stop if UAV GPS not updated over {} frames'.format(opt.patience))

        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))

    def load_image(self, impath, rotate, bins):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_resize, h_resize = int(self.resize[0]*self.scale), int(self.resize[1]*self.scale)
        w_new, h_new = process_resize(w, h, (w_resize, h_resize))
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        if rotate:
            angle = bins*self.bin_interval 
            grayim = self.rotate_image(grayim, angle) # angle>0, rotate image counterclockwise
        # w_rotate, h_rotate = grayim.shape[1], grayim.shape[0]
        # scales = (float(w) / float(w_rotate), float(h) / float(h_rotate))
        return grayim

    def next_frame(self, scale, go_next=True, rotate=False, bins=0):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if (self.i==self.max_length):
            return (None, False)
            
        #update image scale
        self.scale = self.scale*scale
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                time.sleep(.001)
                image = self.stream.read()
            else:
                ret, image = self.cap.read()
            if ret is False or image is None:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_resize, h_resize = int(self.resize[0]*self.scale), int(self.resize[1]*self.scale)
            w_new, h_new = process_resize(w, h, (w_resize, h_resize))
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if rotate:
                angle = bins*self.bin_interval
                image = self.rotate_image(image, angle) # angle>0, rotate image counterclockwise
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file, rotate, bins)
        self.i = self.i + 1 if go_next else self.i
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))
    
    def rotate_image(self, mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

        height, width = mat.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def remove_kpts_on_building(features, labels):
    # screen out basemap keypoints belonging to building
    keys = ['keypoints0', 'scores0', 'descriptors0']
    kpts = features['keypoints0'].astype('int')
    scores = features['scores0']
    descriptors = features['descriptors0']

    valid = labels==0
    
    kpts = features['keypoints0'][valid]
    descriptors = ((descriptors.T)[valid]).T
    scores = scores[valid]
    return {'keypoints0':kpts, 'scores0':scores, 'descriptors0':descriptors}


def segment_keypoints(valid, labels, threshold):
    ground = labels==0
    building = labels==1
    
    grounds = np.logical_and(valid, ground)
    buildings = np.logical_and(valid, building)
    
    grounds_sum = sum(grounds) # number of matched non-building keypoints
    buildings_sum = sum(buildings) # number of matched building keypoints
    
    # # if non-building valid num<threshold and building valid>threshold, select matched building else select non-building keypoints for localization
    # if (grounds_sum<threshold and buildings_sum>threshold) or buildings_sum-grounds_sum>threshold/2:
        # return buildings, grounds, False # use buidling kpts for geolcoalization
    # return grounds, buildings, True
    
    if grounds_sum>=threshold:
        if buildings_sum/grounds_sum<3:
            return grounds, buildings, True, (grounds_sum, buildings_sum)
        else:
            return buildings, grounds, False, (grounds_sum, buildings_sum)
    elif buildings_sum>=threshold:
        return buildings, grounds, False, (grounds_sum, buildings_sum)  # use buidling kpts for geolcoalization
    else:
        return valid, None, True, (grounds_sum, buildings_sum)


def update_last_data(satmap_kpts, mask, spindex, bbox, device):
    xmin, ymin, xmax, ymax = bbox
    image0, keypoints0, descriptors0, scores0 = satmap_kpts
    matches = spindex.intersect((xmin, ymin, xmax-1, ymax-1)) # quadtree will include lower right boundary, so -1 to exclude keypoints lying on that boundary

    keypoints0_ = keypoints0[matches]-[xmin, ymin]
    scores0 = scores0[matches]
    descriptors0 = descriptors0[:,matches]
    
    keypoints0 = torch.from_numpy(keypoints0_).float().to(device)
    scores0 = torch.from_numpy(scores0).float().to(device)
    descriptors0 = torch.from_numpy(descriptors0).float().to(device)
    image0 = frame2tensor(image0[ymin:ymax, xmin:xmax], device)
    
    last_data = {'keypoints0':[keypoints0], 'scores0':[scores0], 'descriptors0':[descriptors0], 'image0':image0}
    if mask is not None:
        update_mask = mask[ymin:ymax, xmin:xmax]
        # print(range, update_mask.shape)
        keypoints0_ = keypoints0_.astype('int')
        labels = update_mask[keypoints0_[:,1], keypoints0_[:,0]]
        return last_data, labels
    else:
        return last_data, None