import os
import Image, ImageFont, ImageDraw
import numpy as np
fp = open("ABC.txt", 'r')
tmp_line = fp.readline()
strs = []
while tmp_line != "":
    tmp_str = tmp_line.split("\n")[0]
    print tmp_str
    strs.append(tmp_str)
    tmp_line = fp.readline()
print len(strs)
for i in range(10000,100000):
    print i
    ran_index = np.random.randint(0, 14326)
    ran_w = np.random.randint(0, 15)
    ran_h = np.random.randint(0, 25)
    ran_r = np.random.randint(150, 255)
    ran_g = np.random.randint(150, 255)
    ran_b = np.random.randint(150, 255)
    text = strs[ran_index]
    im = Image.new("RGB", (200, 40), (ran_r, ran_g, ran_b))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(size=20, filename="Xerox.ttf")
    dr.text((ran_w, ran_h), text, font=font, fill="#000000")
    im.save("/home/liuyi/test/character/{0}_{1}.png".format(i, text))
