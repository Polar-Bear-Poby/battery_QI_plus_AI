import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='pascal', n_classes=None):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset, n_classes=n_classes)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False, n_classes=None):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'simple':
        label_colours = pascal_color_map(N=n_classes)
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


# added by loveiori: All functions below shows color map of Pascal VOC2012
def pascal_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def pascal_color_map_viz(fname = None):
    labels = ['B-ground', 'Aero plane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 
            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Dining-Table', 'Dog', 'Horse',
            'Motorbike', 'Person', 'Potted-Plant', 'Sheep', 'Sofa', 'Train', 
            'TV/Monitor', 'Void/Unlabelled']

    nclasses = 21
    row_size = 80
    col_size = 250
    cmap = pascal_color_map()

    """ arrange these 21 classes to 2D matrix with 3 rows and 7 columns"""
    r = 3
    c = 7
    delta = 10
    array = np.empty((row_size*(r+1), col_size*c, cmap.shape[1]), dtype=cmap.dtype)
    fig=plt.figure()
    for r_idx in range(0,r):
        for c_idx in range(0,c):
            i = r_idx *c + c_idx
            array[r_idx*row_size:(r_idx+1)*row_size, c_idx*col_size: (c_idx+1)*col_size, :] = cmap[i]
            x = c_idx*col_size + delta
            y = r_idx*row_size + row_size/2
            s = labels[i]
            plt.text(x, y,s, fontsize=9, color='white')
            print("write {} at pixel (r={},c={})".format(labels[i], y,x))

    array[r*row_size:(r+1)*row_size, :] = cmap[-1]
    x = 3*col_size + delta
    y = r*row_size + row_size/2
    s = labels[-1]
    plt.text(x, y,s, fontsize=9, color='black')
    print("write {} at pixel (r={},c={})".format(labels[i], y,x))
    plt.title("PASCAL VOC Label Color Map")
    plt.imshow(array)
    axis = plt.subplot(1, 1, 1)
    plt.axis('off')
    if fname:
        plt.savefig(fname, dpi=300,bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


def pascal_color_map_viz_1_column():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = pascal_color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

    for i in range(len(cmap)):
        print("color[", i, "]: ", cmap[i])

    plt.imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()


if __name__ == '__main__':
    pascal_color_map_viz_1_column()