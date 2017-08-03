import json
from PIL import Image
import numpy as np
from random import uniform
digit = Image.fromarray(np.array(json.load(open("2.json")))).convert("L")

rotations = np.array([np.asarray(digit.rotate(uniform(-45, 45)), dtype=np.uint8) for i in range(10)])
np.save("rotated.npy", rotations)