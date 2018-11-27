# author：chenhanping
# date 2018/11/18 下午12:56
# copyright ustc sse
# 生成图片数据
from PIL import Image
from PIL import ImageDraw
from prepare_data import generate_poly_seq, generate_index_in_seq
from Bio.PDB.Polypeptide import *
from Bio.PDB import PDBParser
from Bio.PDB import PDBList


site_path = "/Users/chenhanping/data/ppis/train/site/"
non_path = "/Users/chenhanping/data/ppis/train/non/"
val_site_path = "/Users/chenhanping/data/ppis/val/site/"
val_non_path = "/Users/chenhanping/data/ppis/val/non/"
width = 139
height = 139
base_color_dict = {'I': 20, 'G': 30, 'A': 40, 'V': 35, 'L': 45, 'P': 50, 'F': 55,
                   'Y': 100, 'T': 110, 'M': 120, 'C': 130, 'Q': 140, 'W': 150, 'N': 160, 'S': 170,
                   'H': 210, 'K': 200, 'R': 255,
                   'E': 150, 'D': 180}

# 极性数值
green_color_dict = {
                    'I': 5.2, 'G': 9, 'A': 8.1, 'V': 5.9, 'L': 4.9, 'P': 8, 'F': 5.2,
                    'Y': 6.2, 'T': 8, 'M': 5.7, 'C': 5.5, 'Q': 10.5, 'W': 5.4, 'N': 11.6, 'S': 9.2,
                    'H': 10.4, 'K': 11.3, 'R': 10.5,
                    'E': 12.3,  'D': 13
}

# 使用溶解可达表面积数值
blue_color_dict = {
                    'I': 1.81, 'G': 0.881, 'A': 1.181, 'V': 1.645, 'L': 1.931, 'P': 1.468, 'F': 2.228,
                    'Y': 2.368, 'T': 1.525, 'M': 2.034, 'C': 1.461, 'Q': 1.932, 'W': 2.663, 'N': 1.655, 'S': 1.298,
                    'E': 1.862, 'D': 1.587,
                    'H': 2.025, 'K': 2.258, 'R': 2.56
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
    seq_index = index_dict[chain_index]
    for i in range(1, min(width, len(seq)), 2):
        text = ""
        temp = ""
        start = int(seq_index - (i-1) / 2)
        end = int(seq_index + (i-1) / 2 + 1)
        if start < 0:
            text += "0" * (-1*start)
            start = 0
        if end > len(seq):
            temp += "0" * (end - len(seq))
        end = min(len(seq), height)
        text += seq[start:end] + temp
        text_list.append(text)
    return text_list


def get_relative_aa(seq, residue, index_dict)->list:
    lines = list()
    res_id = residue.get_id()
    chain_index = res_id[1]
    # 当前氨基酸在序列中的位置
    seq_index = index_dict[chain_index]
    # 需要获取的行数
    count_line = min(len(seq), height)
    # 一行需要的列数，即覆盖一个位点的序列长度，借鉴滑动窗口，填满整个宽度，
    count_col = min(width, len(seq))
    # 保证是窗口大小是奇数
    if count_col % 2 != 0:
        count_col -= 1

    # 一条记录的字符起始index
    start = seq_index - int((count_col - 1) / 2)
    for i in range(count_line):
        text = ""
        temp = ""

        end = start + count_col
        if start < 0:
            text += "0" * (-1*start)
            start = 0
        if end > len(seq):
            temp += "0" * (end - len(seq))
        if start >= len(seq):
            break
        text += seq[start:end] + temp
        #print(i, text)
        lines.append(text)
        start += 1
    return lines


def get_base_color(aa, seq_count_dict):

    return base_color_dict.get(aa, 0)


def get_green_color(aa):
    max_val = green_color_dict[max(green_color_dict, key=green_color_dict.get)]
    return int(255 * (green_color_dict.get(aa, 0) / max_val))


def get_blue_color(aa):
    max_val = blue_color_dict[max(blue_color_dict, key=blue_color_dict.get)]
    return int(255*(blue_color_dict.get(aa, 0) / max_val))


def get_color(aa, seq_count_dict):
    red = get_base_color(aa, seq_count_dict)
    green = get_green_color(aa)
    blue = get_blue_color(aa)
    color = (int(red), int(green), int(blue))
    return color


def get_seq_count_dict(seq):
    ss = list(seq)
    seqSet = set(ss)
    dict = {}
    for i in seqSet:
        dict[i] = ss.count(i)
    return dict


def get_red(text, dict):
    count = dict.get(text, 0)
    max_count = max(dict, key=dict.get)
    return round(float(count) / float(dict[max_count]), 2)


def draw_image(text_list, seq_count_dict):
    img = Image.new('RGB', (width, height), (0, 0, 0))
    for i in range(len(text_list)):
        text = text_list[i]

        for j in range(len(text)):
            # draw.text((10 * j + start, 10 * i + 5),
            #           text=text[j],
            #           fill=get_color(text[j], seq_count_dict))
            #print(i,j)
            aa = text[j]
            color = get_color(aa, seq_count_dict)
            img.putpixel((j, i), color)
    return img


def generate_image(isVal, seq, id, chain, site_set, index_dict, seq_count_dict):
    for r in chain:
        save_path = ""
        if r.get_id()[1] in site_set:
            save_path = site_path + id + "_" + chain.get_id() + "_" + str(r.get_id()[1]) + ".png"
            text_list = get_relative_aa(seq, r, index_dict[chain.get_id()])
            img = draw_image(text_list, seq_count_dict)
            if isVal:
                save_path = val_site_path + id + "_" +chain.get_id()+"_"+str(r.get_id()[1])+".png"
            img.save(save_path, "png")
        elif r.get_id()[0] == " " and is_aa(r):
            save_path = non_path + id + "_" + chain.get_id() + "_" + str(r.get_id()[1]) + ".png"
            text_list = get_relative_aa(seq, r, index_dict[chain.get_id()])
            img = draw_image(text_list, seq_count_dict)
            if isVal:
                save_path = val_non_path+ id + "_" +chain.get_id()+"_"+str(r.get_id()[1])+".png"
            img.save(save_path, "png")


if __name__ == '__main__':
    import os
    import random
    pdb_path = "/Users/chenhanping/Downloads/pdb/pdb"
    p = PDBParser()
    pdb = PDBList()
    data_set_path = "/Users/chenhanping/Downloads/PETs-master/dataset/NACCESS/Dset186/"
    file_list = os.listdir(data_set_path)
    i = random.randint(0, len(file_list))
    val_protein = file_list[i]
    print(val_protein)
    count = 0
    for file in file_list:
        filename, _ = os.path.splitext(file)
        print(filename)
        id, chain_id = filename.split("_")
        itf_file = open(os.path.join(data_set_path, file))
        # 读取文件所有行，并将每一行的\n去掉
        lines = [x.rstrip("\n") for x in itf_file.readlines()]
        # 获取最后一行，并得到位点set
        site_line = lines[-1]
        print(id, chain_id)

        site_arr = [int(x) for x in site_line.split(" ")[5: -1] if x.isdigit()]
        site_set = set(site_arr)
        print(site_set)
        if not os.path.exists(pdb_path+id+".ent"):
            pdb.retrieve_pdb_file(id, pdir="/Users/chenhanping/Downloads/pdb/", file_format="pdb")

        if len(chain_id) > 1:
            continue
        s = p.get_structure(id, pdb_path+id+".ent")
        seq_dict = generate_poly_seq(s)
        seq = seq_dict[chain_id]
        print("总长度", len(seq), "位点个数", len(site_set))
        chain = s[0][chain_id]
        index_dict = generate_index_in_seq(s)
        seq_count_dict = get_seq_count_dict(seq)
        isVal = False
        if file == val_protein:
            isVal = True
        generate_image(isVal, seq, id, chain, site_set, index_dict, seq_count_dict)

