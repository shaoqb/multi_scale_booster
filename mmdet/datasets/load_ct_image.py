import os
import cv2
import numpy as np
from scipy.ndimage.morphology import binary_opening

def load_ct_img(config, img_info):

    filename = img_info['filename']
    spacing = img_info['spacing']
    slice_intervals = img_info['slice_intervals']
    do_clip = config.img_do_clip
    num_slice = config.num_slices * config.num_images_3dce

    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    im, mask = load_multislice_img_16bit_png(config, filename, slice_intervals, do_clip, num_slice)

    im = windowing(im, config.windowing)

    if do_clip:  # clip black border
        crop = get_range(mask, margin=0)
        im = im[crop[0]:crop[1] + 1, crop[2]:crop[3] + 1, :]
        # mask = mask[c[0]:c[1] + 1, c[2]:c[3] + 1]
        # print im.shape
    else:
        crop = [0, im.shape[0]-1, 0, im.shape[1]-1]

    im_shape = im.shape[0:2]
    if spacing is not None and config.norm_spacing > 0:  # spacing adjust, will overwrite simple scaling
        im_scale = float(spacing) / config.norm_spacing
    else:
        im_scale = float(config.SCALE) / float(np.min(im_shape))  # simple scaling

    max_shape = np.max(im_shape)*im_scale
    if max_shape > config.max_size:
        im_scale1 = float(config.max_size) / max_shape
        im_scale *= im_scale1

    if im_scale != 1:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, im_scale, np.array(crop).astype('float')


def load_multislice_img_16bit_png(config, filename, slice_intervals, do_clip, num_slice):
    data_cache = {}
    def _load_data(filename, delta=0):
        filename1 = get_slice_name(config, filename, delta)
        if filename1 not in data_cache.keys():
            data_cache[filename1] = cv2.imread(fullpath(filename1, config), -1)
            if data_cache[filename1] is None:
                print('file reading error:', filename1)
        return data_cache[filename1]

    im_cur = _load_data(filename)

    mask = get_mask(im_cur) if do_clip else None

    if config.slice_intervals == 0 or np.isnan(slice_intervals) or slice_intervals < 0:
        ims = [im_cur] * num_slice  # only use the central slice

    else:
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = float(config.slice_intervals) / slice_intervals
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:  # required slice_intervals is a divisible to the actual slice_intervals, don't need interpolation
            for p in range(int((num_slice-1)/2)):
                im_prev = _load_data(filename, - rel_pos * (p + 1))
                im_next = _load_data(filename, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range(int((num_slice-1)/2)):
                intv1 = rel_pos*(p+1)
                slice1 = _load_data(filename, - np.ceil(intv1))
                slice2 = _load_data(filename, - np.floor(intv1))
                im_prev = a * slice1 + b * slice2  # linear interpolation

                slice1 = _load_data(filename, np.ceil(intv1))
                slice2 = _load_data(filename, np.floor(intv1))
                im_next = a * slice1 + b * slice2

                ims = [im_prev] + ims + [im_next]

    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32, copy=False)-32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit

    return im, mask


def get_slice_name(config, filename, delta=0):
    if delta == 0: return filename
    delta = int(delta)
    dirname, slicename = filename.split('//')
    key_slice = int(slicename[:-4])
    filename1 = '%s%s%03d.png' % (dirname, '/', key_slice + delta)

    while not os.path.exists(fullpath(filename1, config)):  # if the slice is not in the dataset, use its neighboring slice
        # print 'file not found:', filename1
        delta -= np.sign(delta)
        filename1 = '%s%s%03d.png' % (dirname, '/', key_slice + delta)
        if delta == 0: break

    return filename1


def fullpath(filename, config):
    filename_full = os.path.join(config.image_path, filename)
    return filename_full


def windowing(im, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

# backward windowing
def windowing_rev(im, win):
    im1 = im.astype(float)/255
    im1 *= win[1] - win[0]
    im1 += win[0]
    return im1

def get_mask(im):
    # use a intensity threshold to roughly find the mask of the body
    th = 32000  # an approximate background intensity value
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
    # mask = binary_dilation(mask)
    # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

    if mask.sum() == 0:  # maybe atypical intensity
        mask = im * 0 + 1
    return mask.astype(dtype=np.int32)

def get_range(mask, margin=0):
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return u, d, l, r


def map_box_back(boxes, crop_x=0, crop_y=0, im_scale=1.):
    boxes /= im_scale
#     boxes[:, [0,2]] += crop_x
#     boxes[:, [1,3]] += crop_y
    boxes[:, 0:-1:2] += crop_x
    boxes[:, 1::2] += crop_y
    return boxes




