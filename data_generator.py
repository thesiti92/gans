#generates datasets of geometrically transformed lines.
#commented out code is for generating randomely rotated twos
import json
from PIL import Image, ImageDraw
import numpy as np
from random import uniform

# load 2 picture from json
# digit = Image.fromarray(np.array(json.load(open("2.json")))).convert("L")

# create black and white halved picture
# pic = Image.fromarray(np.hstack((np.full((28*2,14*2), 255), np.zeros((28*2,14*2))))).convert("L")

# rotations = np.array([np.asarray(pic.rotate(uniform(-90, 90)).crop((14,14,14*3,14*3)), dtype=np.float32) for i in range(8000)])
# np.save("rotated.npy", rotations)

# for i in range(10):
#     pic.rotate(uniform(-45, 45)).crop((14,14,14*3,14*3)).save("sample_%d.jpg" % i)

def moved_lines(samples = 8000):
    out = []
    for i in range(samples):
        im = Image.new('1', (28,28))
        draw = ImageDraw.Draw(im)
        size = uniform(0, 28)
        draw.line([(size, 0), (size, 28)], width=6, fill=255)
        del draw
        out.append(np.asarray(im))
    np.save("moved.npy", np.array(out))

def rotate_lines():
    im = Image.new('1', (28,28))

    draw = ImageDraw.Draw(im)
    draw.line([(0, 0), (28, 28)], width=6, fill=255)
    del draw

    rotations = np.array([np.asarray(im.rotate(uniform(0, 360)), dtype=np.float32) for i in range(8000)])
    np.save("rotated.npy", rotations)

    
def scale_lines(samples = 8000):
    out = []
    for i in range(samples):
        im = Image.new('1', (28,28))
        draw = ImageDraw.Draw(im)
        scale = uniform(1, 14)
        draw.line([(14, 14+scale), (14, 14-scale)], width=6, fill=255)
        del draw
        out.append(np.asarray(im).astype(np.float32))
    np.save("resized.npy", np.array(out))

if __name__ == "__main__":
    length_lines()


# for i in range(10):
#     im.rotate(uniform(0, 360)).save("sample_%d.jpg" % i)
