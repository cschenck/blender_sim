#!/usr/bin/env python
# Borrowed from Daniel Gordon

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BORDER = 5
FONT = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 20)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

def subplot(plots, rows, cols, border=BORDER, titles=None, canvasWidth=IMAGE_WIDTH, canvasHeight=IMAGE_HEIGHT):
    returnedImage = np.full((
        (canvasHeight + 2 * border) * rows,
        (canvasWidth + 2 * border) * cols,
        3), .75, dtype=np.float32)
    for row in xrange(rows):
        for col in xrange(cols):
            if col + cols * row >= len(plots):
                return returnedImage
            im = plots[col + cols * row].astype(np.float32)
            imgMax = np.max(im)
            imgMin = np.min(im)
            im = (im - imgMin) / max((imgMax - imgMin), 0.0001)
            if len(im.shape) < 3:
                im = 255 - (im * 255).astype(np.uint8)
                im = cv2.applyColorMap(im, cv2.COLORMAP_JET).astype(np.float32) / 255
            if im.shape != (canvasHeight, canvasWidth, 3):
                im = cv2.resize(
                        im, (canvasWidth, canvasHeight),
                        interpolation=cv2.INTER_NEAREST)
            if (titles != None and len(titles) > 0 and
                    len(titles) > col + cols * row):
                im *= 255
                im = im.astype(np.uint8)
                im = Image.fromarray(im)
                draw = ImageDraw.Draw(im)
                for x in xrange(9,12):
                    for y in xrange(9, 12):
                        draw.text((x, y), titles[col + cols * row], (0,0,0),
                                font=FONT)
                draw.text((10, 10), titles[col + cols * row], (255,255,255),
                        font=FONT)
                im = np.array(im)
                im = np.array(im).astype(np.float32) / 255.0

            returnedImage[
                    border + (canvasHeight + border) * row : \
                            (canvasHeight + border) * (row + 1),
                    border + (canvasWidth + border) * col : \
                            (canvasWidth + border) * (col + 1),:] = \
                            im
    return returnedImage

# BBoxes are [x1, y1, x2, y2]
def clip_bbox(bboxes, minClip, maxXClip, maxYClip):
    bboxesOut = np.array(bboxes)
    addedAxis = False
    if len(bboxesOut.shape) == 1:
        addedAxis = True
        bboxesOut = bboxesOut[:,np.newaxis]
    bboxesOut[[0,2],:] = np.clip(bboxesOut[[0,2],:], minClip, maxXClip)
    bboxesOut[[1,3],:] = np.clip(bboxesOut[[1,3],:], minClip, maxYClip)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    return bboxesOut

# BBoxes are [x1 y1 x2 y2]
def drawRect(image, bbox, padding, color):
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    bbox = np.round(np.array(bbox)) # mostly just for copying
    bbox = clip_bbox(bbox, padding, imageWidth - padding, imageHeight - padding)
    image[bbox[1]-padding:bbox[3]+padding+1,
            bbox[0]-padding:bbox[0]+padding+1] = color
    image[bbox[1]-padding:bbox[3]+padding+1,
            bbox[2]-padding:bbox[2]+padding+1] = color
    image[bbox[1]-padding:bbox[1]+padding+1,
            bbox[0]-padding:bbox[2]+padding+1] = color
    image[bbox[3]-padding:bbox[3]+padding+1,
            bbox[0]-padding:bbox[2]+padding+1] = color
    return image



# [x1 y1, x2, y2] to [xMid, yMid, width, height]
def xyxy_to_xywh(
        bboxes, clipMin=0, clipWidth=IMAGE_WIDTH, clipHeight=IMAGE_HEIGHT,
        round=False):
    addedAxis = False
    bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    x1 = bboxes[0,:]
    y1 = bboxes[1,:]
    x2 = bboxes[2,:]
    y2 = bboxes[3,:]
    bboxesOut[0,:] = (x1 + x2) / 2
    bboxesOut[1,:] = (y1 + y2) / 2
    bboxesOut[2,:] = x2 - x1
    bboxesOut[3,:] = y2 - y1
    bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,:] = bboxes[4:,:]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut)
    return bboxesOut


# [xMid, yMid, width, height] to [x1 y1, x2, y2]
def xywh_to_xyxy(
        bboxes, clipMin=0, clipWidth=IMAGE_WIDTH, clipHeight=IMAGE_HEIGHT,
        round=False):
    addedAxis = False
    bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    xMid = bboxes[0,:]
    yMid = bboxes[1,:]
    width = bboxes[2,:]
    height = bboxes[3,:]
    bboxesOut[0,:] = xMid - width / 2
    bboxesOut[1,:] = yMid - height / 2
    bboxesOut[2,:] = xMid + width / 2
    bboxesOut[3,:] = yMid + height / 2
    bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,:] = bboxes[4:,:]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut)
    return bboxesOut
