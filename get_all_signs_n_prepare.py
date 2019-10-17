import glob, os, random
import numpy as np
from imageio import imread, imwrite
from scipy.misc import imresize

random.seed(7)

cropped_sign_dir = "D:\\Bhaia_Works\\VIPCUP_2017\\cropped_signs\\*\\*"

all_dir = glob.glob(cropped_sign_dir)

classes = sorted(list(set([os.path.basename(i) for i in all_dir])))

    
data_signs = {}
for c in classes:
    data_signs[c] = []
for i in all_dir:
    data_signs[os.path.basename(i)] += glob.glob(i+"\\*")

resized_signs_dir =  "D:\\Bhaia_Works\\VIPCUP_2017\\resized_signs\\"
save_new = False
if save_new:
    for key in data_signs.keys():
        if not os.path.exists(os.path.join(resized_signs_dir,key)):
            os.makedirs(os.path.join(resized_signs_dir,key))
        for c, i in enumerate(data_signs[key]):
            img = imread(i)
            re_img = imresize(img, (256,256), interp = 'cubic')
            imwrite(os.path.join(
                    os.path.join(resized_signs_dir,key),"{0:03d}.jpg".format(c)),re_img)
        
all_imgs = [] 
all_labels = []   
for c in classes:        
    all_imgs += glob.glob(os.path.join(resized_signs_dir,c)+"\\*")
    all_labels += [int(c)-1]*len(glob.glob(os.path.join(resized_signs_dir,c)+"\\*"))

d_n_l = list(zip(all_imgs,all_labels))
random.shuffle(d_n_l)
train = d_n_l[0:int(len(d_n_l)*0.8)]
valid = d_n_l[int(len(d_n_l)*0.8):]
train_imdir, train_label = zip(*train)
valid_imdir, valid_label = zip(*valid)