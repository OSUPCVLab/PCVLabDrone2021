import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# --- VISUALIZATION ---

def make_localization_plot(GeoLoc, image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, size, center, points, img_box,
                            text, path=None, show_keypoints=False, margin=10, opencv_display=False,
                            opencv_title='', small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    # H, W = max(H0, H1), W0 + margin + int(np.sqrt(size[0]**2+size[1]**2)+1)
    H, W = max(H0, H1), W0 + margin + W1

    # combine image0 and image1
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:W0+margin+W1] = image1
    out = np.stack([out]*3, -1)

    if GeoLoc:
        # # box and center
        cx, cy = center
        out = cv2.circle(out, (cx, cy), 15, (0, 0, 255), 15)
        out = cv2.circle(out, (cx, cy),10, (255, 255, 255), 10)
        out = cv2.circle(out, (cx, cy), 5, (0, 0, 255), 5)
        # plot matching box
        out = cv2.polylines(out,[points],True,(0,0,0),7, cv2.LINE_AA)
        for i in range(4):
            out = drawline(out,pt1=points[i],pt2=points[(i+1)%4])

    # keypoints
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
    
    # matched points
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        # cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 # color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 4, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 4, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (0, 0, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def drawline(img,pt1,pt2,color=(0,0,255),thickness=5,gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    for p in pts:
        cv2.circle(img,p,thickness,color,-1)
    return img


def make_center_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, size, center, points, 
                    text, path=None, show_keypoints=False, margin=10, opencv_display=False,
                    opencv_title='', small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    # combine image0 and image1
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)
    
    # # box and center
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)] #blue, green, yellow and red
    if center is not None:
        for i in range(4):
            cx, cy = center[i]
            out = cv2.circle(out, (cx, cy), 9, colors[i], 9)
            # out = cv2.circle(out, (cx, cy), 6, (255, 255, 255), 6)
            # out = cv2.circle(out, (cx, cy), 3, colors[i], 3)
            # out = cv2.polylines(out,[points],True,(0,255,0),2, cv2.LINE_AA)
    else:
        print('Don\'t have enough matched keypoint pairs, relocalizing...')

    # keypoints
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
    
    # matched points
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        # cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 # color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)
    
    # # plot matched points center
    # cx, cy = (np.mean(mkpts0, axis=0)).astype(int)
    # out = cv2.circle(out, (cx, cy), 9, (0, 0, 255), 9)
    # out = cv2.circle(out, (cx, cy), 6, (255, 255, 255), 6)
    # out = cv2.circle(out, (cx, cy), 3, (0, 0, 255), 3)
    

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out