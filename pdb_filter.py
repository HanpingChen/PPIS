# author：chenhanping
# date 2018/11/19 上午10:05
# copyright ustc sse

# 过滤pdb
from Bio.PDB import *
import os

pdb_path = "/Users/chenhanping/Downloads/pdb/"
json_path = "json/"
p = PDBParser()
pdb_file_list = os.listdir(pdb_path)
count = 0
p = PDBParser()
json_file_list = os.listdir(json_path)
for file in json_file_list:
    file_path = os.path.join(json_path, file)
    if os.path.isfile(file_path):
        (filename, extention) = os.path.splitext(file)
        if extention != ".json":
            continue
        id = filename
        if "pdb"+id+".ent" not in pdb_file_list:
            print(id)
            count += 1
            os.remove(file_path)
# for file in pdb_file_list:
#     file_path = os.path.join(pdb_path, file)
#     if os.path.isfile(file_path):
#         (filename, extention) = os.path.splitext(file)
#         if extention != ".ent":
#             # os.remove(file_path)
#             continue
#         if filename.startswith("pdb"):
#             structure_id = filename[3:]
#             # print(structure_id, filename)
#
#         else:
#             structure_id = filename
#         # structure = p.get_structure(structure_id, file_path)
#         # resolution = structure.header['resolution']
#         # if resolution is None or resolution > 2.5:
#         #     print(structure_id)
#         #     count += 1
#         #     os.remove(file_path)
#         json_file = json_path + structure_id+".json"
#         if not os.path.exists(json_file):
#             print(structure_id)
#             count += 1


print(count)