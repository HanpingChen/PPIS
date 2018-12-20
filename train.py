# 训练模型
import random
src_path = "E:\\image_data\\full_train_data\\"
des_path = "E:\\image_data\\"
from image_process.randon_select_image import *
from mymodel.model_v1 import *


def cross_train(n):
    for i in range(n):
        # 随机生成总样本数和正负样本的比例
        # 样本总数从3万到20万不等
        num_sample = random.randint(30000, 200000)
        # 比例从0.3：1到1：1
        size = random.uniform(0.2, 1)
        size = round(size, 2)
        print(size)
        # 生产样本
        # train_path = random_train_data(src_path, des_path, size, num_sample)
        # if train_path == "":
        #     continue
        model = ModelInception()
        model.train("E:\\image_data\\2018-11-23-16-12-50_100052_0.6845512900705879", epoch=50)


if __name__ == '__main__':
    model = ModelInception()
    model.train("E:\\image_data\\2018-11-23-16-12-50_100052_0.6845512900705879", epoch=120)