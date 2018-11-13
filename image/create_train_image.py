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
        img = Image.new("P", (512, 512), 0)  # 普通八位模式
        draw = ImageDraw.Draw(img)
        for item in dict.items():
            chain_id = item[0]
            chain_seq = item[1]
            break
        with open(base_json_path+structure_id+".json", "r") as f:
            load_dict = json.load(f)
        chain_site = load_dict[chain_id]
        # 绘制序列
        x = -10
        y = 5
        for i in range(len(chain_seq)):
            x = x + 10
            if x > 500:
                x = 0
                y = y + 10
            draw.text((x, y), chain_seq[i], fill=255)
        img.save(base_img_path + structure_id + ".png", "png")
        #img.show()
        # 绘制label
        label = Image.new("P", (512, 512), 0)
        draw = ImageDraw.Draw(label)
        site_index_set = set()
        for item in chain_site:
            site_index_set.add(item['seq_index'])
        x = -10
        y = 5
        for i in range(len(chain_seq)):
            x = x + 10
            if x > 500:
                x = 0
                y = y + 10
            for k in range(x, x + 5):
                for l in range(y, y + 5):
                    if i in site_index_set:
                        label.putpixel((k, l), 255)
                    else:
                        label.putpixel((k, l), 0)
        label.save(base_label_path + structure_id + ".png", "png")