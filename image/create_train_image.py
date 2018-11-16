from PIL import Image
from Bio.PDB import PDBParser
from PIL import ImageDraw
from PIL import ImageFont
from prepare_data import generate_poly_seq
import os
import json
p = PDBParser()
base_img_path = "E:\\PPIS\\train_data\\image\\"
base_label_path = "E:\\PPIS\\train_data\\label\\"
base_pdb_path = "E:\\PPIS\\pdb\\"
base_json_path = "E:\\PPIS\\train_data\\json\\"

hy_dict = {
    'R': -4.5, 'K': -3.9, 'N': -3.5, 'D': -3.5, 'Q': -3.5, 'E': -3.5, 'H': -3.2, 'P': -1.6,
    'Y': -1.3, 'W': -0.9, 'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8, 'M': 1.9, 'C': 2.5,
    'F': 2.8, 'L': 3.8, 'V': 4.2, 'I': 4.5
}


def get_pix_value(aa):
    return int(-28*hy_dict[aa]+126)

import math
count = 0
file_list = os.listdir(base_json_path)
for file in file_list:
    file_path = os.path.join(base_json_path, file)
    if os.path.isfile(file_path):
        filename, extention = os.path.splitext(file)
        count = count + 1
        if count == 201:
            break
        if filename.startswith("pdb"):
            structure_id = filename[3:]
        else:
            structure_id = filename
        protein_path = os.path.join(base_pdb_path, "pdb"+structure_id+".ent")
        structure = p.get_structure(structure_id, protein_path)
        dict = generate_poly_seq(structure)
        length = 0
        for item in dict.items():
            length += len(dict[item[0]])
            break
        width = (int(math.sqrt(length * 25)) + 1)
        height = (int(math.sqrt(length * 25)) + 1)
        img = Image.new("P", (width, height), 0)  # 普通八位模式
        draw = ImageDraw.Draw(img)
        for item in dict.items():
            chain_id = item[0]
            chain_seq = item[1]
            break
        with open(base_json_path+structure_id+".json", "r") as f:
            load_dict = json.load(f)
        chain_site = load_dict[chain_id]
        # 绘制序列
        x = -5
        y = 0
        for i in range(len(chain_seq)):
            x = x + 5
            if x > width -6:
                x = 0
                y = y + 5
                if y > height - 6:
                    break
            # img.putpixel((x, y), get_pix_value(chain_seq[i % (len(chain_seq))]))
            for k in range(x, x + 5):
                for l in range(y, y + 5):
                    img.putpixel((k, l), get_pix_value(chain_seq[i]))
        img = img.resize((256, 256))
        img.save(base_img_path + structure_id + ".png", "png")
        #img.show()
        # 绘制label
        label = Image.new("P", (width, height), 0)
        draw = ImageDraw.Draw(label)
        site_index_set = set()
        for item in chain_site:
            site_index_set.add(item['seq_index'])
        x = -5
        y = 0
        for i in range(len(chain_seq)):
            x = x + 5
            if x > width-6:
                x = 0
                y = y + 5
                if y > height-6:
                    break

            for k in range(x, x + 5):
                for l in range(y, y + 5):
                    if i in site_index_set:
                        label.putpixel((k, l), 255)
                    else:
                        label.putpixel((k, l), 0)
        label = label.resize((256, 256))
        label.save(base_label_path + structure_id + ".png", "png")
