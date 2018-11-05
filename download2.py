from Bio.PDB import PDBList

pdb = PDBList()
f = open("pdblist.txt", 'r')
pdb_id_list = f.readlines()
import time
import os
for i in range(len(pdb_id_list)):
    pdb_id_list[i] = pdb_id_list[i].rstrip('\n')
count = 0
for i in range(17000, 18000):
    pdb.retrieve_pdb_file(pdb_id_list[i], file_format="pdb", pdir="pdb")
    count = count + 1
    if not os.path.exists("pdb\\pdb"+pdb_id_list[i].lower()+".ent"):
        time.sleep(0.5)
    print("完成 ", count, " 个下载")