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

def gen_lines():
    im = Image.new('1', (28,28))

    draw = ImageDraw.Draw(im)
    draw.line([(0, 0), (28, 28)], width=6, fill=255)
    del draw

    rotations = np.array([np.asarray(im.rotate(uniform(0, 360)), dtype=np.float32) for i in range(8000)])
    np.save("rotated.npy", rotations)

# for i in range(10):
#     im.rotate(uniform(0, 360)).save("sample_%d.jpg" % i)
