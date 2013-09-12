from skimage.color import rgb2gray
from skimage.data import imread
import numpy as np
import scipy as sp
from colors import color
from skimage import data
import sys


if __name__=="__main__":
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        img = imread(image_file, as_grey=True)
    else:
        print "no image specified. running demo"
        img = sp.misc.lena()

    img = img[::10, ::10] + img[1::10, ::10] + img[::10, 1::10] + img[1::10, 1::10]
    img_gray = img

    black = 232
    white = 255

    norm = (img_gray - np.min(img_gray)) / (np.max(img_gray) -np.min(img_gray))
    norm = (white - black) * norm + black
    norm = norm.astype(np.int)

    for i, row in enumerate(norm):
        items = []
        for item in row:
            x = u"\u25A0"
            x = u"\u25CF"
            print color(x, fg=item),
            items.append(color(x, fg=item))
        #print "".join(items)
        print

