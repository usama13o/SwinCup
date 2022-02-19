import nibabel as nib
import numpy as np
import os
from utils.util import mkdir
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz",'png','tiff','jpg',"bmp"])

def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image
def open_target_np_slides(path):
    im = open_image(path)
    mask= np.array(im)
    li = (np.unique(mask))
    if 29 in li:
        mask[mask==29]=0
    # normal case
    # if len(li)>5:
    if 'fixed' in path:
       # print('found normal slide' + path)
        mask[mask!=255]=0
        mask[mask==255]=2
    #tumour
    else:
       # print('found tumour slide' + path)
        mask = ~mask
        mask[mask!=255]=0

        mask[mask==255]=1
    li = (np.unique(mask,return_counts=True))
    # print(li)
    return mask[:,:,0,np.newaxis]
def open_target_np(path):
    im = open_image(path)
    mask= np.array(im)
    li = (np.unique(mask))
    if 29 in li:
        mask[mask==29]=0
    # normal case
    if len(li)>5:
        mask[mask!=255]=0
        mask[mask==255]=2
    #tumour
    else:
        mask[mask!=255]=0
        mask[mask==255]=1
    return mask[:,:,0,np.newaxis]
def open_target_np_peso(path):
    im = open_image(path)
    mask= np.array(im)
    mask[mask==1]=0
    mask[mask==2]=1
    return mask[:,:,0,np.newaxis]

def open_target_np_glas(path):
    im = open_image(path)
    mask= np.array(im)
    mask[mask!=0]=1
    return mask[:,:,np.newaxis]

color2index = {
    (0, 64, 128) : 0,
    (64, 128, 0) : 1,
    (243, 152, 0) : 2,
    (255, 255, 255) : 3,
}

def rgb2mask(img):

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1) 
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])] 
        except:
            pass
    return mask
def open_wss_target(path):
    im = open_image(path)
    mask = np.array(im)
    # mask =rgb2mask(mask)

    return mask[:,:,np.newaxis].astype(np.uint8)
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array

def open_image_np_bw(path):
    im = open_image(path)
    array = np.array(im)
    return array[:,:,np.newaxis].repeat(3,axis=2)
def load_nifti_img(filepath, dtype):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta


def write_nifti_img(input_nii_array, meta, savedir):
    mkdir(savedir)
    affine = meta['affine'][0].cpu().numpy()
    pixdim = meta['pixdim'][0].cpu().numpy()
    dim    = meta['dim'][0].cpu().numpy()

    img = nib.Nifti1Image(input_nii_array, affine=affine)
    img.header['dim'] = dim
    img.header['pixdim'] = pixdim

    savename = os.path.join(savedir, meta['name'][0])
    print('saving: ', savename)
    nib.save(img, savename)


def check_exceptions(image, label=None):
    if label is not None:
        if image.shape[:-1] != label.shape[:-1]:
            print('Error: mismatched size, image.shape = {0}, '
                  'label.shape = {1}'.format(image.shape, label.shape))
            #print('Skip {0}, {1}'.format(image_name, label_name))
            raise(Exception('image and label sizes do not match'))

    if image.max() < 1e-6:
        print('Error: blank image, image.max = {0}'.format(image.max()))
        #print('Skip {0} {1}'.format(image_name, label_name))
        raise (Exception('blank image exception'))

    if label.max() < 1e-6:
        print('Error:  label blank, image.max = {0}'.format(label.max()))
        #print('Skip {0} {1}'.format(image_name, label_name))
        raise (Exception('blank label exception'))