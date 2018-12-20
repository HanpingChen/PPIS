# 下采样
non_path = "../data/ppis/train/non"
small_train = "../data/ppis/small_train/non"
import os
import shutil
imgs = os.listdir(non_path)
import random
select = random.sample(range(len(imgs)), 15000)
for i in select:
    shutil.copy(os.path.join(non_path, imgs[i]), os.path.join(small_train,imgs[i]))
