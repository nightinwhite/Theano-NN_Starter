#coding:utf-8
import os
from PIL import Image
import numpy as np
def get_date_sum  (file_path) :
    files = os.listdir(file_path)
    return len(files)

def get_date (file_path):
    images = []
    anss = []
    sum_d = get_date_sum(file_path)
    for i in range(sum_d):
        print "{0}/{1}".format(i, sum_d)
        i, a = get_date_i(file_path, i)
        images.append(i)
        anss.append(a)
    return images, anss

def get_date_i  (file_path , i) :
    files = os.listdir(file_path)
    images = []
    ans_s = []
    image_shape = (200, 60)
    image = np.asarray(Image.open(file_path+"/"+files[i]).convert('L').resize(image_shape), np.float32)
    image = image.transpose((1, 0))
    image = np.array([image], dtype=np.float32)
    ans = files[i].split('_')[1]
    ans = ans.split('.')[0]
    ans = [ord(c)-64 for c in ans]
    for c in ans :
        if c >58 :
            return get_date_i(file_path, (i+1) % len(files))
    ans = fill_blank(ans, blank_index=0, fill_len=93)
    return image, ans
def get_ans_maxlen (file_path):
    max_len = 0
    max_i = -1
    max_ans = ""
    files = os.listdir(file_path)
    i = 0
    for f in files :
        ans = f.split('_')[1]
        ans = ans.split('.')[0]
        tmp_len = len(ans)
        if tmp_len > max_len:
            max_len = tmp_len
            max_i = i
            max_ans = ans
        i += 1
    return max_len, max_i, max_ans

def fill_blank  (arr, blank_index,fill_len):
    a_len = len(arr)
    return [blank_index if i % 2 == 0 or i/2 >= a_len else arr[i/2] for i in range(fill_len)]

#'/home/liuyi/test/captcha'
#print get_ans_maxlen('/home/liuyi/test/captcha')
# image,f = get_date_i('/home/liuyi/test/captcha',277)
# os.remove('/home/liuyi/test/captcha'+'/'+f)