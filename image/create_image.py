# author：chenhanping
# date 2018/11/5 下午2:54
# copyright ustc sse
from PIL import Image
from Bio.PDB import PDBParser
from PIL import ImageDraw
from PIL import ImageFont
from prepare_data import generate_poly_seq
p = PDBParser()
hy_dict = {
    'R': -4.5, 'K': -3.9, 'N': -3.5, 'D': -3.5, 'Q': -3.5, 'E': -3.5, 'H': -3.2, 'P': -1.6,
    'Y': -1.3, 'W': -0.9, 'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8, 'M': 1.9, 'C': 2.5,
    'F': 2.8, 'L': 3.8, 'V': 4.2, 'I': 4.5
}


def get_f(aa):
    return int(-28*hy_dict[aa]+126)


structure_id = "1aby"
clean_file_name = "E:\PPIS\pdb\pdb"+structure_id+".ent"
structure = p.get_structure(structure_id, clean_file_name)
dict = generate_poly_seq(structure)
print(dict)
img = Image.new("P", (256, 256), 0)#RGB模式
draw = ImageDraw.Draw(img)
import json
with open("../json/1aby.json","r") as f:
    load_dict = json.load(f)

chain_a_site = load_dict['A']
print(chain_a_site)
site_count_dict = {}
for site in chain_a_site:
    if site['seq'] in site_count_dict.keys():
        site_count_dict[site['seq']] = site_count_dict[site['seq']] + 1
    else:
        site_count_dict[site['seq']] = 1

print(site_count_dict)
count_dict = {}
print(len(dict['A']))
font = ImageFont.load_default()
chain_a = dict['A']
site_index_set = set()
for item in chain_a_site:
    site_index_set.add(item['seq_index'])
for c in chain_a:
    if c in count_dict.keys():
        count_dict[c] = count_dict[c] + 1
    else:
        count_dict[c] = 1
print(count_dict)
for keys in count_dict.keys():
    if keys not in site_count_dict.keys():
        site_count_dict[keys] = 0
x = -5
y = 0
for i in range(len(chain_a)):
    x = x + 5
    if x > 249:
        x = 0
        y = y + 5
    for k in range(x, x + 5):
        for l in range(y, y + 5):

            img.putpixel((k, l), int(get_f(chain_a[i])))
# for i in range(2000):
#     for j in range(500):
#         r, g, b = img.getpixel((i, j))
#         if r != 0 and g != 0 and b != 0:
#             img.putpixel((i, j), (120, 230, 130))
img.show()
base_img_path = "E:\\PPIS\\train_data\\image\\"
label = Image.new("P", (256, 256), 0)
draw = ImageDraw.Draw(label)

img.save(base_img_path+structure_id+".png", "png")
x = -5
y = 0
for i in range(len(chain_a)):
    x = x + 5
    if x > 249:
        x = 0
        y = y + 5
    for k in range(x, x + 5):
        for l in range(y, y + 5):
            if i in site_index_set:
                label.putpixel((k, l), 255)
                print(k, l)
            else:
                label.putpixel((k, l), 0)

base_label_path = "E:\\PPIS\\train_data\\label\\"
label.save(base_label_path+structure_id+".png", "png")
label.show()