import cv2, glob, os
from imageio import imread

DATA_DIR = "D:\\Bhaia_Works\\VIPCUP_2017\\data\\*"
LABEL_DIR = "D:\\Bhaia_Works\\VIPCUP_2017\\labels\\*"

data_videos =  glob.glob(DATA_DIR)
label_files =  glob.glob(LABEL_DIR)

signs_savedir = "D:\\Bhaia_Works\\VIPCUP_2017\\cropped_signs\\" 
imgs_savedir = "D:\\Bhaia_Works\\VIPCUP_2017\\data_image_split\\" 

data_n_label_zip = list(zip(data_videos, label_files))


for video, file in data_n_label_zip:
    
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 1
    is_dir = imgs_savedir + video.split(os.sep)[-1].split('.')[0]
    if not os.path.exists(is_dir):
        os.makedirs(is_dir)
        
    while success:
      cv2.imwrite(is_dir + "\\{0:03d}.jpg".format(count), image)    
      success,image = vidcap.read()
      count += 1
      
      
    f = open(file , "r")
    if f.mode == 'r':
        contents = f.readlines()
        f.close()  
    ls_dir = signs_savedir + video.split(os.sep)[-1].split('.')[0]   
    if not os.path.exists(ls_dir):
        os.makedirs(ls_dir)
    
    prev_img = None   
    for c, v in enumerate(contents[1:]):
        t = v.split('_')
        img_name = t[0] 
        sign_type = t[1]
        sdir = os.path.join(ls_dir, sign_type)
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        if not img_name == prev_img:
            img = imread(os.path.join(is_dir,img_name)+'.jpg')
            prev_img = img_name
        top_left_x = min(map(int,t[2::2]))
        top_left_y = min(map(int,t[3::2]))
        bot_right_x = max(map(int,t[2::2]))
        bot_right_y = max(map(int,t[3::2]))
        
        cropped_img = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
        cv2.imwrite(sdir + "\\{0:03d}.jpg".format(c+1), cropped_img)



