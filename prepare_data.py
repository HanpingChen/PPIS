# author：chenhanping
# date 2018/11/non 下午12:26
# copyright ustc sse

from Bio.PDB import PDBParser
from Bio.PDB import Select
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import *
from Bio.PDB.DSSP import *
from util.protein_util import *

import os
from Bio.Data.SCOPData import protein_letters_3to1
import json


class ResidueSelect(Select):
    """
    剔除不正常的残基，包括水分子等等
    """
    def accept_residue(self, residue):
        if residue.get_id()[0] == ' ':
            return True
        else:
            return False


class ChainSelect(Select):
    """
    提取出复合物中的指定单链
    """

    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return True
        else:
            return False


def process_complex(structure):
    """
    处理复合物，去除里面的水分子，并保存一些有用的头信息
    :param structure:
    :return:
    """
    io = PDBIO()
    io.set_structure(structure)
    if not os.path.exists("clean_data"):
        os.makedirs("clean_data")
    io.save("clean_data/"+structure.get_id()+".ent", ResidueSelect())


def generate_poly_seq(structure)->dict:
    """
    提取出复合物的所有肽链的序列
    :param structure:
    :return: dict
    """
    ppb = PPBuilder()
    poly_seq_dict = {}
    model = structure[0]
    for chain in model:
        seq = ""
        for pp in ppb.build_peptides(chain):
            seq += str(pp.get_sequence())
        poly_seq_dict[chain.get_id()] = seq
    return poly_seq_dict


def generate_index_in_seq(structure, chain_id_list)->dict:
    """
    获取残基在链上的index对应在多肽序列上的的index
    :param chain_id_list:
    :param structure:
    :return: {A:{non:10}} 表示在A链序列上，第一个残基在A链上的位置是10
    """
    model = structure[0]
    dict = {}
    for id in chain_id_list:
        chain = model[id]
        chain_dict = {}
        i = 0
        for r in chain:
            if r.get_id()[0] == ' ' and is_aa(r):
                chain_dict[r.get_id()[1]] = i
                i = i + 1
        dict[chain.get_id()] = chain_dict
    return dict


def calculate_site(structure, filename):
    """
    计算一个复合物中的结合位点
    :param structure:
    :return:
    """
    # 首先提取出结构中的所有多肽链，并写成pdb文件
    model = structure[0]

    # 将数据写入到json中
    site_dict = {}
    io = PDBIO()
    # 获取蛋白质组成信息
    compound = structure.header['compound']
    # 获取两个compound中的各一个链
    chain_id_list = list()
    chain_id_list.append(compound['1']['chain'].split(",")[0].upper())
    chain_id_list.append(compound['2']['chain'].split(",")[0].upper())
    print(chain_id_list)
    # 获取seq的position和chain id的映射字典
    index_dict = generate_index_in_seq(structure, chain_id_list)
    for id in chain_id_list:
        chain = model[id]
        io.set_structure(structure)
        if not os.path.exists("chain"):
            os.makedirs("chain")
        io.save("chain/"+structure.get_id()+"_"+chain.get_id()+".ent", ChainSelect(chain.get_id()))
    # 提取出所有多肽链的序列，写成dictionary
    # 获取最大acc字典:
    max_acc_dict = residue_max_acc['Sander']
    # 计算dssp
    # step1,分别计算复合物和所有单链的dssp,
    # 计算复合物
    complex_dssp_dict = dssp_dict_from_pdb_file(filename, DSSP="./mkdssp")[0]
    # 计算所有的链的dssp
    for id in chain_id_list:
        chain = model[id]
        # 记录每一个链中的位点
        site_list = list()
        chain_file_name = "chain/"+structure.get_id()+"_"+chain.get_id()+".ent"
        chain_dict = dssp_dict_from_pdb_file(chain_file_name, DSSP="./mkdssp")[0]
        # dict的形式如
        # ('A', (' ', non, ' ')): ('M', '-', 108, 360.site, -92.4, non, site, site.site, 2, -site.4, site, site.site, 127, -site.non)
        # step2 ,遍历chain，判断残基是否占最大可达面积的5%
        for item in chain_dict.items():
            residue_index = item[0][1][1]
            residue = chain[residue_index]
            # 获取这个残基的acc
            acc = float(item[1][2])
            # 获取这个残基可以达到的最大acc
            max_acc = max_acc_dict[residue.get_resname()]
            if acc / max_acc >= 0.05:
                # step3, 比较当前的acc和在复合物中的acc的差值是否大于1
                # 获取复合物中的当前残基的acc, item[site]是复合物中dict的key，也就是residue的唯一标识
                # 获取到的dict形式与上述的dict形式一致，所以结果元组的第三位就是acc
                complex_acc = complex_dssp_dict[item[0]][2]
                if acc - complex_acc >= 1:
                    # 这就是位点了
                    site_list.append(residue_index)
        site_dict[structure_id+"_"+chain.get_id()] = site_list
    # 写入到txt文件
    # 删除中间文件
    # 删除chain
    for chain_id in chain_id_list:
        os.remove("chain/"+structure.get_id()+"_"+chain_id+".ent")
    return site_dict, index_dict, chain_id_list


def start(structure_id, filename, save_directory):
    # structure_id = "1lky"
    # filename = "data/"+structure_id+".ent"
    # 判断是否已经解析过
    p = PDBParser()
    structure = p.get_structure(structure_id, filename)
    dict, index_dict, chain_id_list = calculate_site(structure, filename)
    count = 0
    # 位点
    site_directory = os.path.join(save_directory, "site_set")
    # index_dict
    index_dict_directory = os.path.join(save_directory, "index_dict")
    # 蛋白质序列
    protein_seq_directory = os.path.join(save_directory, "protein_seq")
    paths = [protein_seq_directory, site_directory, index_dict_directory]
    for path in paths:
        if not os.path.exists(path):
            print('creating directory', path)
            os.makedirs(path)
    model = structure[0]
    for item in dict.items():
        site_list = item[1]
        count = count + len(site_list)
        # 保存位点数据
        f = open(os.path.join(site_directory, item[0]+".txt"), 'w')
        site_str = ",".join([str(x) for x in site_list])
        print(item[0], site_str)
        f.write(site_str)
    # 保存index_dict 数据
    for chain_id, d in index_dict.items():
        with open(os.path.join(index_dict_directory, structure_id + "_" + chain_id+".json"), 'w') as f:
            json.dump(d, f)
    # 保存protein seq 数据
    seq_dict = generate_poly_seq(structure)
    for chain_id in chain_id_list:
        seq = seq_dict[chain_id]
        f = open(os.path.join(protein_seq_directory, structure_id + "_" + chain_id+".txt"), 'w')
        f.write(seq)
    print("共找到", count, "个位点")

    #print(json_data)
    #for chain in model:


if __name__ == '__main__':
    import time
    s = time.time()
    with open("config/path_config.json", 'r') as f:
        paths = json.load(f)
    base_path = paths['pdb_path']
    save_directory = paths['data_directory']
    file_list = os.listdir(base_path)
    count = 0
    for file in file_list:
        file_path = os.path.join(base_path, file)
        if os.path.isfile(file_path):
            (filename, extention) = os.path.splitext(file)
            count = count + 1
            if filename.startswith("pdb"):
                structure_id = filename[3:]
                print(structure_id, filename)
                #start()
            else:
                structure_id = filename
            try:
                start(structure_id, file_path, save_directory)
            except Exception as e:
                print(filename, str(e))
                continue
            print(count, "/", len(file_list))
    e = time.time()
    print("耗时", e - s)
