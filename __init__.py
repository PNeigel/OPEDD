import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import os

from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte

img_per_package = 50
n_packages = 21
this_file_path = Path(os.path.abspath(__file__)).parent

all_img_results = [0] * n_packages

baseline = 119.958
focal = 1407.8

def loadImg(number):
    '''
    Returns: (left_img, right_img, depth_img) from the dataset with image-number 'number'
    The Left and Right Images have 4 channels. This is how the ZED camera returns them.
    The 4th channel is the alpha channel and just filled with all 255s.
    '''
    
    assert isinstance(number, int) | isinstance(number, float), "Number must be int or float."
    assert number in range(1018), "Only values in [0, 1017] allowed."
    number = int(number)
    
    package_no = number // img_per_package
    
    left_img = io.imread(this_file_path / f"package_{package_no}/{number}.png")
    right_img = io.imread(f"C:/Files/Datasets/dfki_outdoor/right/{number}.png")
    depth_img = np.load(f"C:/Files/Datasets/dfki_outdoor/depth/{number}.npy")
    disp_img = toDisp(depth_img)
    
    return left_img, right_img, depth_img, disp_img
    
def getNext():
    for i in range(1018):
        yield loadImg(i)
        
class Polygon():
    
    vertices = []
    image_shape = None
    
    def __init__(self, vertices, image_shape):
        self.vertices = vertices
        self.image_shape = image_shape
        
    def mask(self):
        image = Image.new("L", self.image_shape[::-1]) #Greyscale image unsigned 32bit
        draw = ImageDraw.Draw(image)
        draw.polygon(self.vertices, fill=255)
        return np.array(image)
    
    def __str__(self):
        return f"Polygon of {len(self.vertices)} vertices"
    
    def __repr__(self):
        return self.__str__()
    
class PersonPolygon(Polygon):
    
    ID = None
    
    def __init__(self, vertices, ID, image_shape):
        self.ID = ID
        super().__init__(vertices, image_shape)
        
    def __int__(self):
        return self.ID

def extractPolysFromJSON(json_path, root=None):
    '''
    Extracts the polygons from a dataset json-file.
    Since there are 50 images per json, this functions returns a list with 50 elements.
    Every list-element is a tuple (img_path, polygons), where polygons is a list with the polygons
    appearing in the image.
    '''
    
    with open(json_path, "rt") as f:
        import json
        info = json.load(f)
    images = info['_via_img_metadata']
    root = Path(info['_via_settings']['core']['default_filepath']) if root is None else root
    package_img_results = []
    for img_key in images:
        img_polygons = []
        img_filename = images[img_key]['filename']
        img_path = root / img_filename
        
        # Get original image shape
        img_og = io.imread(str(img_path))
        height, width, channels = img_og.shape
        
        regions = images[img_key]['regions']
        for region in regions:
            # Get vertices list and person ID
            shape_attributes = region['shape_attributes']
            region_attributes = region['region_attributes']
            points_x = shape_attributes['all_points_x']
            points_y = shape_attributes['all_points_y']
            vertices = list(zip(points_x, points_y))
            try:
                ID = int(region_attributes['Person_id'])
            except ValueError as e:
                print(f"{str(img_path)} has an invalid id!")
                raise e
            img_polygons.append(PersonPolygon(vertices, ID, (height, width)))
        
        package_img_results.append((img_path, img_polygons))

    sortfunc = lambda x: int(x[0].with_suffix('').name)
    return sorted(package_img_results, key=sortfunc)

def getPolygons(number):
    '''
    Returns the polygon list of image with the given number.
    Will cache the results of a package if called for the first time.
    '''
    
    global all_img_results
    
    package_no = number // img_per_package
    if all_img_results[package_no] == 0:
        print(f"Reading package {package_no}. Result will be cached for faster access of images in the same package.")
        package_list = extractPolysFromJSON(this_file_path / f"package_{package_no}/package_{package_no}.json", this_file_path / f"package_{package_no}")
        all_img_results[package_no] = package_list
    return all_img_results[package_no][number % img_per_package][1]
    
def bboxFromMask(mask):
    y_nonzero, x_nonzero = mask.nonzero()
    y_max = y_nonzero.max()
    y_min = y_nonzero.min()
    x_max = x_nonzero.max()
    x_min = x_nonzero.min()
    return [(x_min, y_min), (x_max, y_max)]

def genPersonMasks(poly_list):
    if len(poly_list) == 0:
        return [],[]
    ids = np.unique(list(map(int, poly_list)))
    height, width = poly_list[0].image_shape
    masks = []
    cumulative = np.zeros((height, width), dtype=np.uint8)
    for Id in ids:
        id_masks = np.asarray([poly.mask() for poly in poly_list if int(poly)==Id])
        id_mask_combined = np.where(np.any(id_masks, axis=0), 255, 0)
        id_mask_combined = np.where(cumulative, 0, id_mask_combined)
        masks.append(id_mask_combined)
        cumulative = np.where(id_mask_combined, 255, cumulative)
    if -1 in ids:
        for i in range(1, len(masks)):
            masks[i] = np.where(masks[0] > 0, 0, masks[i])
        return masks[1:], list(map(bboxFromMask, masks[1:]))
    else:
        return masks, list(map(bboxFromMask, masks))
    
def getPatchGen(number, masked=False, noNan=False, noInf=False, reshape=None):
    '''
    Returns a generator that yields all the patches in the image with the number as a normal or masked array.
    Params:
        * masked: If set to true, the patches are masked such that only values in the ground truth segmentation mask are viable -> masked array.
        * noNan: If set to true, NaN values in the depth image are masked -> masked array.
        * noInf: If set to true, inf values in the depth image are masked -> masked array.
        If none of these parameters are set, a normal numpy array is returned.

        * reshape: If a tuple shape is given, the image will be resized to that shape before applying mask and cutting patches.

        Returns:
            * Left image patch
            * Right image patch
            * Patch Depth
            * Patch disparity
            * Top Left Patch Coordinate (x1,y1)
            * Bottom Right Patch Coordinate (x2,y2)
            * Mask
        (left_i, right_i, depth_i, disp_i), (x1,y1), (x2,y2), mask_i.copy()
    '''
    left, right, depth, disp = loadImg(number) # left, right: uint; depth, disp: float32

    if reshape is not None:
        assert len(reshape) == 2, "Reshape must be 2-tuple"
        height_new, width_new = reshape
        height_old, width_old, n_channels = left.shape
        left = img_as_ubyte(resize(left, reshape))
        right = img_as_ubyte(resize(right, reshape))
        depth = resize(depth, reshape).astype(np.float32)
        # Attention: Resized depth can lose a lot of information, if nan's and inf's are
        # still present in the image.
        disp = resize(disp, reshape).astype(np.float32)

    polygons = getPolygons(number)
    masks, bboxes = genPersonMasks(polygons)
    
    for i in range(len(masks)):

        (x1,y1),(x2,y2) = bboxes[i]

        if reshape is not None:
            mask_i = resize(masks[i], reshape)
            mask_i = img_as_ubyte(mask_i/mask_i.max())
            x1 = int(1.*x1 * width_new / width_old)
            x2 = int(1.*x2 * width_new / width_old)
            y1 = int(1.*y1 * height_new / height_old)
            y2 = int(1.*y2 * height_new / height_old)
        else:
            mask_i = masks[i].copy()

        mask_cutoff = 10

        if masked:
            rgb_mask = np.tile((mask_i<=mask_cutoff), (4,1,1))
            rgb_mask = np.moveaxis(rgb_mask, 0, 2)
            left_i = np.ma.masked_array(left, rgb_mask, copy=True)
            right_i = np.ma.masked_array(right, rgb_mask, copy=True)
        else:
            left_i = left.copy()
            right_i = right.copy()
        if masked or noNan or noInf:
            depth_mask = ((mask_i<=mask_cutoff) & masked) | (np.isinf(depth) & noNan) | (np.isnan(depth) & noInf)
            depth_i = np.ma.masked_array(depth, depth_mask, copy=True)
            disp_i = np.ma.masked_array(disp, depth_mask, copy=True)
        else:
            depth_i = depth.copy()
            disp_i = disp.copy()


        left_i = left_i[y1:y2, x1:x2]
        right_i = right_i[y1:y2, x1:x2]
        depth_i = depth_i[y1:y2, x1:x2]
        disp_i = disp_i[y1:y2, x1:x2]

        yield (left_i, right_i, depth_i, disp_i), (x1,y1), (x2,y2), mask_i.copy()

def toDepth(disparity, scale=1.0):
    '''
    Takes as input a disparitymap and returns the depthmap, according to the focal length and baseline of the dataset.
    '''
    depth = disparity.astype(np.float32).copy()
    w = ~(np.isnan(disparity) | (disparity == 0)) # Only values not nan or zero
    depth[w] = focal*baseline*scale/depth[w]
    w = np.isinf(disparity) # Where disparity is infinite... (should not happen)
    depth[w] = np.nan # ...depth is something between 0 and nan
    w = (disparity == 0) # Where disparity is zero...
    depth[w] = np.inf # ... depth is infinite
    return depth

def toDisp(depth, scale=1.0):
    '''
    Takes as input a depthap and returns the disparitymap, according to the focal length and baseline of the dataset.
    '''
    disparity = depth.copy()
    w = ~(np.isnan(depth) | (depth == 0)) # Only values not nan or zero
    disparity[w] = focal*baseline*scale/depth[w] 
    w = np.isinf(depth) # Where depth is infinite...
    disparity[w] = 0.0 # ...disparity is zero
    w = depth == 0 # Where depth is zero...
    disparity[w] = np.inf

    return disparity