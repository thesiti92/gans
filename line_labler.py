import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


angles = [np.degrees(cv2.HoughLines(cv2.Canny(sample,50,150,apertureSize = 3),1,np.pi/180,10)[0][0][1]) for sample in (np.load("y_gen.npy")*255).astype(np.uint8).reshape(-1,28,28)]

hist = np.histogram(angles)
fig = plt.figure()

plt.bar(hist[1][:-1], hist[0])
fig.savefig("histogram.png")

# for i, sample in enumerate((np.load("x_gen.npy")*255).astype(np.uint8).reshape(-1,28,28)[:100]):
#     edges = cv2.Canny(sample,50,150,apertureSize = 3)
#     color = cv2.cvtColor(sample,cv2.COLOR_GRAY2BGR)

#     lines = cv2.HoughLines(edges,1,np.pi/180,10)
#     for rho,theta in lines[0]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))

#         cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)

#     cv2.imwrite('houghlines%d.jpg' % i,color)

