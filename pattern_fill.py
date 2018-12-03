import sys
import math
from PIL import Image
import numpy as np

def main(path, input_width, input_height):
    input_width = int(input_width)
    input_height = int(input_height)
    
    im = Image.open(path)
    width, height = im.size
    
    horizontal_num = math.ceil(input_width / width)
    vertical_num = math.ceil(input_height / height)
    
    a = range(0, horizontal_num)
    lst = [path for i in a]
    
    images = map(Image.open, lst)
    widths, heights = zip(*(i.size for i in images))

    new_im = Image.new('RGB', (input_width, input_height))

    x_offset = 0
    y_offset = 0
    images0 = map(Image.open, lst)

    for x in range(0, horizontal_num):
        for y in range(0, vertical_num):
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
        y_offset += im.size[1]
        x_offset = 0

    img_array = np.array(new_im)

    return img_array
    # new_im.save('cropped.jpg')

if __name__ == '__main__':
    path = sys.argv[1]
    input_width = sys.argv[2]
    input_height = sys.argv[3]
    main(path, input_width, input_height)

