# author：chenhanping
# date 2018/11/5 下午2:54
# copyright ustc sse
from PIL import Image
from Bio.PDB import PDBParser
from PIL import ImageDraw
from PIL import ImageFont
from prepare_data import generate_poly_seq
p = PDBParser()

structure_id = "1aby"
clean_file_name = "../clean_data/"+structure_id+".ent"
structure = p.get_structure(structure_id, clean_file_name)
dict = generate_poly_seq(structure)
print(dict)
img = Image.new("RGB", (512, 512), (0, 0, 0))#RGB模式
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
x = -10
y = 5
for i in range(len(chain_a)):
    x = x + 10
    if x > 500:
        x = 0
        y = y + 10
    draw.text((x, y), chain_a[i], fill=(count_dict[chain_a[i]]+100,site_count_dict[chain_a[i]]*10+50, count_dict[chain_a[i]]+50))
# for i in range(2000):
#     for j in range(500):
#         r, g, b = img.getpixel((i, j))
#         if r != 0 and g != 0 and b != 0:
#             img.putpixel((i, j), (120, 230, 130))
img.show()

label = Image.new("P", (512, 512), 0)
draw = ImageDraw.Draw(label)

x = -5
y = 0
for i in range(len(chain_a)):
    x = x +5
    if x > 500:
        x = 0
        y = y + 10
    for k in range(x, x + 5):
        for l in range(y, y + 5):
            if i in site_index_set:
                label.putpixel((k, l), 255)
            else:
                label.putpixel((k, l), 0)


label.show()