import json
from PIL import Image
import numpy as np
from random import uniform
digit = Image.fromarray(np.array(json.load(open("2.json")))).convert("L")


pic = Image.fromarray(np.hstack((np.full((28*2,14*2), 255), np.zeros((28*2,14*2))))).convert("L")

# rotations = np.array([np.asarray(digit.rotate(uniform(-45, 45))) for i in range(10)])
# np.save("rotated.npy", rotations)

for i in range(10):
    pic.rotate(uniform(-45, 45)).crop((14,14,14*3,14*3)).save("sample_%d.jpg" % i)

