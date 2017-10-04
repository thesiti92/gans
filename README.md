# Gans!
## Exploring how Geometric Transforms Map onto the Z-Space Vector

My code for a small GANs research project I did at HT Kung's lab the summer of 2017. I initially implemented a DCGAN in Torch, but Chainer was already configured on the GPU we had, so I ported it over. The purpose of the research project was to see how a geometric transformation in a picture maps onto the noise vector after training. 

### 1 dimensional Z-space of a DCGAN trained on randomely rotated lines:
![alt text](https://raw.githubusercontent.com/thesiti92/gans/master/sample_pics/1d_100e.png)

---

### 1 dimensional Z-space mappings during training on same dataset:
![alt text](https://raw.githubusercontent.com/thesiti92/gans/master/sample_pics/1d_100e_progression.png)

---

### 1 dimensional Z-space mappings during training on dataset of resized lines:
![alt text](https://raw.githubusercontent.com/thesiti92/gans/master/sample_pics/1d_100e_length.png)

---

### Final 1 dimensional Z-space mappings of DCGAN trained on dataset of resized and rotated lines:
![alt text](https://raw.githubusercontent.com/thesiti92/gans/master/sample_pics/1d_100e_2transforms.png)

---

### Final 2 dimensional Z-space mappings of DCGAN trained on dataset of resized and rotated lines:
![alt text](https://raw.githubusercontent.com/thesiti92/gans/master/sample_pics/2d_100e_both.png )

