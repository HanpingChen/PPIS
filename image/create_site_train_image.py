# author：chenhanping
# date 2018/11/16 上午7:43
# copyright ustc sse
from prepare_data import generate_poly_seq, generate_index_in_seq
from Bio.Data.SCOPData import protein_letters_3to1 as t2o
import os
import json
from Bio.PDB.Polypeptide import *
from PIL import Image
from PIL import ImageDraw
from Bio.PDB import PDBParser

# 设置需要的位点数量
max_site_count = 1000
# 设置非位点数量
max_non_site_count = 1000
site_count = 0
non_site_count = 0

width = 150
height = 150

site_path = "../data/train/site/"
non_path = "../data/train/non/"

base_color_dict = {'I': 20, 'G': 30, 'A': 40, 'V': 35, 'L': 45, 'P': 50, 'F': 55,
                   'Y': 100, 'T': 110, 'M': 120, 'C': 130, 'Q': 140, 'W': 150, 'N': 160, 'S': 170,
                   'H': 210, 'K': 200, 'R': 255,
                   'E': 150, 'D': 180}
green_color_dict = {
                    'I': 131, 'G': 75, 'A': 89, 'V': 177, 'L': 131, 'P': 115, 'F': 165,
                    'Y': 181, 'T': 119, 'M': 149, 'C': 121, 'Q': 146, 'W': 204, 'N': 132, 'S': 105,
                    'H': 155, 'K': 146, 'R': 174,
                    'E': 147,  'D': 133}

blue_color_dict = {
                    'I': 1, 'G': 1, 'A': 1, 'V': 1, 'L': 1, 'P': 1, 'F': 1,
                    'Y': 2, 'T': 2, 'M': 2, 'C': 2, 'Q': 2, 'W': 2, 'N': 2, 'S': 2,
                    'E': 3, 'D': 3,
                    'H': 4, 'K': 4, 'R': 4
}


def get_text(seq, residue, index_dict)->list:
    """

    :param index_dict:
    :param seq:
    :param residue:
    :return:
    """
    text_list = list()
    res_id = residue.get_id()
    chain_index = res_id[1]
    print()
    seq_index = index_dict[chain_index]
    for i in range(1, 12, 2):
        text = ""
        temp = ""
        start = int(seq_index - (i-1) / 2)
        end = int(seq_index + (i-1) / 2 + 1)
        print(start, end)
        if start < 0:
            text += "0" * (-1*start)
            start = 0
        if end > len(seq):
            temp += "0" * (end - len(seq))
            end = len(seq)
        text += seq[start:end] + temp
        text_list.append(text)
    return text_list


def get_base_color(aa):
    return base_color_dict.get(aa, 0)


def get_green_color(aa):
    return green_color_dict.get(aa, 0)


def get_blue_color(aa):
    return int(255*(blue_color_dict.get(aa, 0) / .4))


def get_color(aa):
    red = get_base_color(aa)
    green = 150
    blue = 80
    color = (red, green, blue)
    return color


def create_image(id, seq, chain, index_dict, site_dict):
    """
    为一个氨基酸创建图像
    如果是位点，则存放于site_path,否则存放于non_path
    根据滑动窗口的理论，我们选取这个氨基酸相邻的1，3，5，7，9，11 个氨基酸写入图像
    :param index_dict: 残基位置和多肽序列的index之间的映射字典
    :param seq:
    :param chain:
    :return:
    """
    count = 0
    site_set = set()
    dict_arr = site_dict[chain.get_id()]
    for item in dict_arr:
        site_set.add(item['chain_index'])
    for r in chain:
        if not is_aa(r):
            continue

        text_list = get_text(seq, r, index_dict[chain.get_id()])
        print(text_list)
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        i = 0
        for text in text_list:
            i = i + 1
            for j in range(len(text)):
                draw.text((10*j+5, 10*i + 5), text=text[j], fill=get_color(text[j]))
        save_path = ""
        if count in site_set:
            save_path = site_path + id + "_" + chain.get_id()+"_" + str(count) + ".png"
        else:
            save_path = non_path + id + "_" + chain.get_id()+"_"+ str(count) + ".png"
        img.save(save_path, "png")
        count += 1


if __name__ == '__main__':
    p = PDBParser()
    id = "1aby"
    file = "../data/1aby.pdb"
    s = p.get_structure(id, file)
    model = s[0]
    chain = model['B']
    dict = generate_poly_seq(s)
    index_dict = generate_index_in_seq(s)
    with open("../json/1aby.json", "r") as f:
        load_dict = json.load(f)
    create_image(id, dict['B'], chain, index_dict, load_dict)