#coding:utf-8
import os
from PIL import Image
import numpy as np
def get_date_sum  (file_path) :
    files = os.listdir(file_path)
    return len(files)

# def get_date (file_path):
#     images = []
#     anss = []
#     sum_d = get_date_sum(file_path)
#     for i in range(sum_d):
#         print "{0}/{1}".format(i, sum_d)
#         i, a = get_date_i(file_path, i)
#         images.append(i)
#         anss.append(a)
#     return images, anss

def get_date(file_path, ans_name):
    fp = open(file_path+"/"+ans_name, 'r')
    tmp_line = fp.readline()
    images = []
    anss = []
    refer =np.concatenate([np.arange(26), np.full((6,), 26, dtype=np.int32), np.arange(26)])
    #blank_index = 26
    #print refer
    m = 0
    while tmp_line != "":
        print m
        m+=1
        tmp_lines = tmp_line.split(":")
        tmp_file = tmp_lines[0]
        tmp_ans = tmp_lines[1]
        image_shape = (200, 60)
        tmp_path = file_path + "/" + tmp_file
        image = np.asarray(Image.open(tmp_path).convert('L').resize(image_shape), np.float32)
        image = (image.transpose((1, 0))+0.0)/255
        image = np.array([image], dtype=np.float32)
        images.append(image)
        tmp_ans = [refer[ord(tmp_ans[i])-65] if ord(tmp_ans[i])-65 >= 0 and ord(tmp_ans[i])-65 <= 57 else 26 for i in range(len(tmp_ans))]
        tmp_ans = fill_blank(tmp_ans, 26, 41 )
        anss.append(tmp_ans)
        tmp_line = fp.readline()
    images = np.asarray(images)
    anss = np.asarray(anss)
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
    ans = fill_blank(ans, blank_index=26, fill_len=41)
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


def CTC_B(A):
    blank_num = 0
    i = len(A) - 1
    j = i
    while i != 0:
        j = i - 1
        if A[i] != blank_num and A[j] == A[i]:
            del A[i]
        elif A[i] == blank_num:
            del A[i]
        i -= 1
    if A[0] == blank_num:
        del A[0]
    return A

def date_difference(y, out):
    out = np.argmax(out, axis=2)
    y = y.tolist()
    out = out.tolist()
    s = len(y)
    right_sum = 0
    for i in range(s):
        y_s = [y[i][m] for m in range(len(y[i]))]
        out_s = [out[i][m] for m in range(len(out[i]))]
        y_i = CTC_B(y[i])
        out_i = CTC_B(out[i])
        #print"{0} \nVS\n {1}\n".format(y_s, out_s)
        if len(y_i)!=len(out_i):
            continue
        else:
            isright = True
            for j in range(len(y_i)):
                if y_i[j]!=out_i[j]:
                    isright = False
                    break
            if isright:
                right_sum+=1
    return right_sum
#'/home/liuyi/test/captcha'
#print get_ans_maxlen('/home/liuyi/test/captcha')
# image,f = get_date_i('/home/liuyi/test/captcha',277)
# os.remove('/home/liuyi/test/captcha'+'/'+f)