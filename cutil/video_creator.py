#!/usr/bin/env python

import os
import cv2
import numpy as np
import subprocess
import tempfile

import connor_util as cutil

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow 
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head 
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head 
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

class VideoCreator:
    def __init__(self, width, height):
        self.width_ = width
        self.height_ = height
        self.frames = np.zeros((1,height,width,3), dtype=np.uint8)
        self.shift = 0

    def length(self):
        return self.frames.shape[0]+self.shift

    def width(self):
        return self.frames.shape[2]

    def height(self):
        return self.frames.shape[1]

    def save(self, out_fp, codec='XVID', fps=30):
        writer = cv2.VideoWriter(out_fp, cv2.cv.CV_FOURCC(*codec), fps, (self.width(), self.height()))
        for t in range(self.length()):
            writer.write(self.frames[t,...])
        writer.release()

    def saveMP4(self, out_fp, fps=30):
        tmp = tempfile.NamedTemporaryFile()
        self.save(tmp.name, fps=fps)
        command = "avconv -i %s -c:v libx264 -c:a copy %s" % (tmp.name, out_fp)
        subprocess.call(command.split())

    def saveGif(self, out_fp, fps=30):
        import matplotlib.pyplot as plt 
        import matplotlib.animation as animation
     
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax = fig.add_axes([0,0,1.0,1.0])
        ax.set_axis_off()
        fig.tight_layout()
        fig.set_size_inches(self.width()/100.0, self.height()/100.0, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ims = map(lambda i: (ax.imshow(self.frames[i,...,::-1]),ax.set_title('')), range(0, self.frames.shape[0]))
        im_ani = animation.ArtistAnimation(fig, ims, interval=1000.0/fps, repeat_delay=0, blit=False)
        #plt.show()
        im_ani.save(out_fp, writer='imagemagick', savefig_kwargs={'bbox_inches':'tight'})


    def savePartial(self, end, out_fp=None, codec='XVID', fps=30, finish=False):
        if out_fp is not None:
            self.writer = cv2.VideoWriter(out_fp, cv2.cv.CV_FOURCC(*codec), fps, (self.width(), self.height()))
        for i in range(self.shift, end):
            self.writer.write(self.frames[i-self.shift,...])
        self.frames = self.frames[(end-self.shift):,...]
        self.shift = end
        if finish:
            self.writer.release()

    def load(self, fp):
        cap = cv2.VideoCapture(fp)
        length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        self.frames = np.zeros((length, height, width, 3), dtype=np.uint8)
        self.width_ = width
        self.height_ = height
        self.shift = 0
        i = 0
        pm = cutil.ProgressMonitor(lambda : 1.0*i/length, update_interval=None)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frames[i,...] = frame
            i += 1
            pm.print_progress()
        pm.stop()

    def crop(self, start=0, end=None, x=0, y=0, w=None, h=None):
        if end is None:
            end = self.length()
        if w is None:
            w = self.width()
        if h is None:
            h = self.height()
        other = VideoCreator(w, h)
        other.setFrames(self.frames[start-self.shift:end-self.shift,y:y+h,x:x+w,...], 0)
        return other

    def playVideo(self, fps=30):
        spin = True
        while spin:
            for i in range(self.frames.shape[0]):
                cv2.imshow("Video", self.frames[i,...])
                k = cv2.waitKey(1000/fps)
                if k in [27, 1048603]:
                    spin = False
                    break
            if spin:
                print("Restarting from the beginning.")

    def __expand_frames(self, start, end):
        if end-self.shift >= self.frames.shape[0]:
            self.frames = np.concatenate((self.frames, 
                  np.zeros((end-self.shift - self.frames.shape[0],self.height_,self.width_,3), dtype=self.frames.dtype)),
                axis=0)

    def solidColor(self, start, end, color):
        self.__expand_frames(start, end)
        for c in range(self.frames.shape[-1]):
            self.frames[(start-self.shift):(end-self.shift),:,:,c] = color[c]

    def __listify(self, x):
        if type(x) in [list, tuple]:
            return x
        else:
            return [x]

    def placeText(self, lines, start, end, location='center', font=cv2.FONT_HERSHEY_COMPLEX, scale=2, 
                    color=(255,255,255), thickness=2, fade_in=None, fade_out=None, x_shift=0, y_shift=0):
        self.__expand_frames(start, end)
        lines = self.__listify(lines)
        font = self.__listify(font)
        scale = self.__listify(scale)
        thickness = self.__listify(thickness)
        fade_in = self.__listify(fade_in)
        fade_out = self.__listify(fade_out)
        x_shift = self.__listify(x_shift)
        y_shift = self.__listify(y_shift)
        if type(color[0]) not in [list, tuple]:
            color = [color]

        sizes = []
        for i,line in enumerate(lines):
            f = font[min(i, len(font)-1)]
            s = scale[min(i, len(scale)-1)]
            t = thickness[min(i, len(thickness)-1)]
            (w,h),b = cv2.getTextSize(line, f, s, t)
            w = int(round(w))
            h = int(round(h))
            b = int(round(b))
            sizes.append((w, h, b))

        if location in ['northwest', 'southwest', 'west']:
            x_coeff = 0
            start_x = 0
        elif location in ['north', 'center', 'south']:
            x_coeff = 0.5
            start_x = self.width_/2
        else:
            x_coeff = 1.0
            start_x = self.width_
        if location in ['northwest', 'northeast', 'north']:
            y = 0
        elif location in ['west', 'center', 'east']:
            y = self.height_/2 - sum([x[1]+x[2] for x in sizes])/2
        else:
            y = self.height_ - sum([x[1]+x[2] for x in sizes])
        y = int(round(y))

        for i,line in enumerate(lines):
            f = font[min(i, len(font)-1)]
            s = scale[min(i, len(scale)-1)]
            t = thickness[min(i, len(thickness)-1)]
            c = color[min(i, len(color)-1)]
            fi = fade_in[min(i, len(fade_in)-1)]
            fi = fi if fi is not None else 0
            fo = fade_out[min(i, len(fade_out)-1)]
            fo = fo if fo is not None else 0
            xs = x_shift[min(i, len(x_shift)-1)]
            ys = y_shift[min(i, len(y_shift)-1)]
            w,h,b = sizes[i]
            y += h
            yy = y + ys
            x = int(round(start_x - x_coeff*w)) + xs
            bjs = x
            bje = x+w
            bis = yy - h
            bie = yy + b
            for j in range(start, start+fi):
                r = 1.0*(j-start)/fi
                orig = self.frames[j-self.shift,bis:bie,bjs:bje,...].copy()
                cv2.putText(self.frames[j-self.shift,...], line, (x,yy), f, s, c, t)
                self.frames[j-self.shift,bis:bie,bjs:bje,...] = r*self.frames[j-self.shift,bis:bie,bjs:bje,...] + (1-r)*orig
            for j in range(start+fi, end-fo):
                cv2.putText(self.frames[j-self.shift,...], line, (x,yy), f, s, c, t)
            for j in range(end-fo, end):
                r = 1.0*(j - (end-fo))/fo
                orig = self.frames[j-self.shift,bis:bie,bjs:bje,...].copy()
                cv2.putText(self.frames[j-self.shift,...], line, (x,yy), f, s, c, t)
                self.frames[j-self.shift,bis:bie,bjs:bje,...] = (1-r)*self.frames[j-self.shift,bis:bie,bjs:bje,...] + r*orig
            y += b

    def drawArrow(self, start, end, p, q, color, arrow_magnitude=9, thickness=1, fade_in=0, fade_out=0):
        self.__expand_frames(start, end)
        for t in range(start, start+fade_in):
            r = 1.0*(t-start)/fade_in
            orig = self.frames[t-self.shift,...].copy()
            draw_arrow(self.frames[t-self.shift,...], p, q, color, arrow_magnitude=arrow_magnitude, thickness=thickness)
            self.frames[t-self.shift,...] = r*self.frames[t-self.shift,...] + (1-r)*orig
        for t in range(start+fade_in, end-fade_out):
            draw_arrow(self.frames[t-self.shift,...], p, q, color, arrow_magnitude=arrow_magnitude, thickness=thickness)
        for t in range(end-fade_out, end):
            r = 1.0*(t - (end-fade_out))/fade_out
            orig = self.frames[t-self.shift,...].copy()
            draw_arrow(self.frames[t-self.shift,...], p, q, color, arrow_magnitude=arrow_magnitude, thickness=thickness)
            self.frames[t-self.shift,...] = (1-r)*self.frames[t-self.shift,...] + r*orig
        

    def append(self, other, crossfade=0):
        self.combine(other, 0, other.length(), self.length() - crossfade, self.length())

    def append_load(self, fp, crossfade=0):
        cap = cv2.VideoCapture(fp)
        length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        start = self.length() - crossfade
        end = start + length
        self.__expand_frames(start, end)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize(frame, self.width(), self.height())
            if i < crossfade:
                r = 1.0*i/crossfade
                self.frames[start+i-self.shift,...] = r*frame + (1-r)*self.frames[start+i-self.shift,...]
            else:
                self.frames[start+i-self.shift,...] = frame
            i += 1

    def combine(self, other, other_start, other_end, self_start, self_end_trans):
        ntrans = (self_end_trans - self_start)
        length = other_end - other_start
        self_end = self_start + length
        self.__expand_frames(self_start, self_end)
        for i in range(ntrans):
            r = 1.0*i/ntrans
            self.frames[self_start+i-self.shift,...] = (r*other.frames[other_start+i-other.shift,...] 
                + (1-r)*self.frames[self_start+i-self.shift,...])
        for i in range(ntrans,length):
            self.frames[self_start+i-self.shift,...] = other.frames[other_start+i-other.shift,...]

    def __resize_params(self, img_width, img_height, width, height):
        rw = 1.0*width/img_width
        rh = 1.0*height/img_height
        x_offset = 0
        y_offset = 0
        # Black bars on the side.
        if rh < rw:
            ratio = rh
            x_offset = int((width - ratio*img_width)/2)
        else:
            ratio = rw
            y_offset = int((height - ratio*img_height)/2)
        return ratio, x_offset, y_offset

    def resize(self, img, width, height):
        if img.shape[1] == width and img.shape[0] == height:
            return img
        ratio, x_offset, y_offset = self.__resize_params(img.shape[1], img.shape[0], width, height)
        img = cv2.resize(img, (int(ratio*img.shape[1]), int(ratio*img.shape[0])))
        ret = np.zeros((height, width, 3), dtype=img.dtype)
        ret[y_offset:(y_offset+img.shape[0]),x_offset:(x_offset+img.shape[1]),:] = img
        return ret


    def loadFrames(self, fps, start):
        end = len(fps) + start
        self.__expand_frames(start, end)
        for i,fp in enumerate(fps):
            img = cv2.imread(fp)
            self.frames[i+start-self.shift,...] = self.resize(img, self.width_, self.height_)

    def setFrames(self, frs, start, heatmap=None):
        self.__expand_frames(start, start+frs.shape[0])
        for t in range(frs.shape[0]):
            img = frs[t,...]
            while len(img.shape) < 3:
                img = np.expand_dims(img, len(img.shape))
            # Only a single color channel.
            if img.shape[2] == 1:
                if heatmap is not None:
                    #img = cv2.applyColorMap(img, heatmap)
                    img = cutil.grayscaleToHeatmap(img, maxVal=255, rgb_max=255)
                else:
                    img = np.tile(img, (1,1,3))
            self.frames[start+t-self.shift,...] = self.resize(img, self.width(), self.height())


    def grid(self, vcs, vcs_ranges, start):
        maxh = max([x.height() for x in vcs.flatten()])
        maxw = max([x.width() for x in vcs.flatten()])
        length = np.max(vcs_ranges[...,1] - vcs_ranges[...,0])
        nrows = vcs.shape[0]
        ncols = vcs.shape[1]
        img = np.zeros((nrows*maxh, ncols*maxw, 3), dtype=np.uint8)
        self.__expand_frames(start, start+length)
        for t in range(length):
            for i in range(nrows):
                for j in range(ncols):
                    if vcs[i,j] is None:
                        continue
                    r1 = vcs_ranges[i,j,0]
                    try:
                        img[(i*maxh):((i+1)*maxh),(j*maxw):((j+1)*maxw),...] = self.resize(vcs[i,j].frames[r1+t,...], maxw, maxh)
                    except:
                        cutil.keyboard("ERROR: video_creator.py:210")
            self.frames[start+t-self.shift,...] = self.resize(img, self.width(), self.height())

    def grid_shift(self, vcs_start, vcs_end, vcs_ranges, start):
        # First let's setup all the variables.
        smaxh = max([x.height() for x in vcs_start.flatten()])
        smaxw = max([x.width() for x in vcs_start.flatten()])
        snrows = vcs_start.shape[0]
        sncols = vcs_start.shape[1]
        emaxh = max([x.height() for x in vcs_end.flatten()])
        emaxw = max([x.width() for x in vcs_end.flatten()])
        enrows = vcs_end.shape[0]
        encols = vcs_end.shape[1]
        length = np.max(vcs_ranges[...,1] - vcs_ranges[...,0])
        height = self.height()
        width = self.width()
        sratio, sx_off, sy_off = self.__resize_params(smaxw*sncols, smaxh*snrows, width, height)
        eratio, ex_off, ey_off = self.__resize_params(emaxw*encols, emaxh*enrows, width, height)
        self.__expand_frames(start, start+length)

        # Next get the parentage.
        parents = vcs_end.copy()
        for i in range(enrows):
            for j in range(encols):
                parents[i,j] = None
                for pi in range(snrows):
                    for pj in range(sncols):
                        if vcs_start[pi, pj] == vcs_end[i, j]:
                            parents[i,j] = (pi, pj)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        for t in range(length):
            img[...] = 0
            for i in range(enrows):
                for j in range(encols):
                    pi, pj = parents[i,j]
                    r1 = vcs_ranges[i,j,0]
                    si1 = pi*smaxh*sratio + sy_off
                    sj1 = pj*smaxw*sratio + sx_off
                    si2 = (pi+1)*smaxh*sratio + sy_off
                    sj2 = (pj+1)*smaxw*sratio + sx_off
                    ei1 = i*smaxh*eratio + ey_off
                    ej1 = j*smaxw*eratio + ex_off
                    ei2 = (i+1)*smaxh*eratio + ey_off
                    ej2 = (j+1)*smaxw*eratio + ex_off
                    r = 1.0*t/length
                    i1 = int(round((1-r)*si1 + r*ei1))
                    i2 = int(round((1-r)*si2 + r*ei2))
                    j1 = int(round((1-r)*sj1 + r*ej1))
                    j2 = int(round((1-r)*sj2 + r*ej2))
                    try:
                        img[i1:i2,j1:j2,...] = self.resize(vcs_end[i,j].frames[t+r1,...], j2-j1, i2-i1)
                    except:
                        cutil.keyboard('err')
            self.frames[start+t-self.shift,...] = self.resize(img, self.width(), self.height())

    def overlay(self, other, start, other_start, other_end, threshold=0):
        length = other_end - other_start
        self.__expand_frames(start, start+length)
        for t in range(length):
            img = other.frames[other_start+t-other.shift,...]
            idxs = np.where((img[...,0] > threshold) | (img[...,1] > threshold) | (img[...,2] > threshold))
            if idxs[0].shape[0] == 0:
                continue
            for c in range(3):
                ii = idxs + (np.array([c]*idxs[0].shape[0]),)
                jj = (np.array([start + t-self.shift]*idxs[0].shape[0]),) + ii
                self.frames[jj] = img[ii]

    def blend(self, other, start, other_start, other_end, other_alpha=None):
        length = other_end - other_start
        self.__expand_frames(start, start+length)
        for t in range(length):
            img = other.frames[other_start+t-other.shift,...]
            under = self.frames[start+t-self.shift,...]
            if other_alpha is not None:
                bf = other_alpha.frames[t,...].max(axis=-1)/255.0
            else:
                bf = img.max(axis=-1)/255.0
            for c in range(3):
                self.frames[start+t-self.shift,...,c] = img[...,c]*bf + under[...,c]*(1 - bf)

    def repeatLastFrame(self, n):
        start = self.length()
        end = self.length() + n
        self.__expand_frames(start, end)
        for t in range(start, end):
            self.frames[t-self.shift,...] = self.frames[start-1-self.shift,...]


