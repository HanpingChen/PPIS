# author：chenhanping
# date 2018/11/non 下午12:26
# copyright ustc sse

from Bio.PDB import PDBParser
from Bio.PDB import Select
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.DSSP import residue_max_acc
from SiteResidue import SiteResidue
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
        poly_seq_dict[chain.get_id()] =seq
    return poly_seq_dict


def generate_index_in_seq(structure)->dict:
    """
    获取残基在链上的index对应在多肽序列上的的index
    :param structure:
    :return: {A:{non:10}} 表示在A链序列上，第一个残基在A链上的位置是10
    """
    model = structure[0]
    dict = {}
    for chain in model:
        chain_dict = {}
        i = 0
        for r in chain:
            if r.get_id()[0] == ' ' and is_aa(r):
                chain_dict[r.get_id()[1]] = i
                i = i + 1
        dict[chain.get_id()] = chain_dict
    return dict


def calculate_site(structure)->dict:
    """
    计算一个复合物中的结合位点
    :param structure:
    :return:
    """
    # 首先提取出结构中的所有多肽链，并写成pdb文件
    model = structure[0]
    # 获取seq的position和chain id的映射字典
    index_dict = generate_index_in_seq(structure)
    # 将数据写入到json中
    json_data = {}
    site_dict = {}
    io = PDBIO()
    for chain in model:
        io.set_structure(structure)
        if not os.path.exists("chain"):
            os.makedirs("chain")
        io.save("chain/"+structure.get_id()+"_"+chain.get_id()+".ent", ChainSelect(chain.get_id()))
    # 提取出所有多肽链的序列，写成dictionary
    poly_seq_dict = generate_poly_seq(structure)
    # 获取最大acc字典:
    max_acc_dict = residue_max_acc['Wilke']
    # 计算dssp
    # step1,分别计算复合物和所有单链的dssp,
    # 计算复合物
    complex_file_name = "clean_data/"+structure.get_id()+".ent"
    complex_dssp_dict = dssp_dict_from_pdb_file(complex_file_name, DSSP="./mkdssp")[0]
    # print(complex_dssp_dict)
    # 计算所有的链的dssp
    for chain in model:
        chain_site_dict_list = list()
        # 记录每一个链中的位点
        site_list = list()
        chain_file_name = "chain/"+structure.get_id()+"_"+chain.get_id()+".ent"
        chain_dict = dssp_dict_from_pdb_file(chain_file_name, DSSP="./mkdssp")[0]
        # dict的形式如
        # ('A', (' ', non, ' ')): ('M', '-', 108, 360.site, -92.4, non, site, site.site, 2, -site.4, site, site.site, 127, -site.non)
        # step2 ,遍历chain，判断残基是否占最大可达面积的16%
        for item in chain_dict.items():
            residue_site_dict = {}
            residue_index = item[0][1][1]
            residue = chain[residue_index]
            # 获取这个残基的acc
            acc = float(item[1][2])
            # 获取这个残基可以达到的最大acc
            max_acc = max_acc_dict[residue.get_resname()]
            if acc / max_acc > 0.16:
                # step3, 比较当前的acc和在复合物中的acc的差值是否大于1
                # 获取复合物中的当前残基的acc, item[site]是复合物中dict的key，也就是residue的唯一标识
                # 获取到的dict形式与上述的dict形式一致，所以结果元组的第三位就是acc
                complex_acc = complex_dssp_dict[item[0]][2]
                if acc - complex_acc > 1:
                    # 这就是位点了
                    site_residue = SiteResidue()
                    site_residue.acc = acc
                    site_residue.chain_id = chain.get_id()
                    site_residue.chain_index = residue_index
                    # site_residue.seq = poly_seq_dict[chain.get_id()][residue_index-non]
                    site_residue.seq = protein_letters_3to1.get(residue.get_resname())
                    site_residue.seq_index = index_dict[chain.get_id()][residue_index]
                    site_list.append(site_residue)
                    residue_site_dict['seq'] = protein_letters_3to1.get(residue.get_resname())
                    residue_site_dict['seq_index'] = index_dict[chain.get_id()][residue_index]
                    residue_site_dict['chain_index'] = residue_index
                    chain_site_dict_list.append(residue_site_dict)
        json_data[chain.get_id()] = chain_site_dict_list
        site_dict[chain.get_id()] = site_list
    # 写入到json文件
    with open("json/"+structure.get_id()+".json","w") as f:
        json.dump(json_data,f)
        print("写入完成")
    print(json_data)
    # 删除中间文件
    # 删除chain
    for chain in model:
        os.remove("chain/"+structure.get_id()+"_"+chain.get_id()+".ent")
    return site_dict


def start(structure_id, filename):
    # structure_id = "1lky"
    # filename = "data/"+structure_id+".ent"
    # 判断是否已经解析过
    if os.path.exists("json/"+structure_id+".json"):
        return
    p = PDBParser()
    structure = p.get_structure(structure_id, filename)
    # 处理蛋复合物
    process_complex(structure)
    clean_file_name = "clean_data/"+structure_id+".ent"
    structure = p.get_structure(structure_id, clean_file_name)
    dict = calculate_site(structure)
    count = 0
    model = structure[0]
    for item in dict.items():
        site_list = item[1]
        count = count + len(site_list)
        # for site in site_list:
        #     print(site)
    print("共找到", count, "个位点")
    # 删除clean_data
    os.remove(clean_file_name)
    #print(json_data)
    #for chain in model:


if __name__ == '__main__':
    import time
    s = time.time()
    base_path = "/Users/chenhanping/Downloads/pdb/"
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
                start(structure_id, file_path)
            except Exception as e:
                print(filename, str(e))
                continue
            print(count, "/", len(file_list))
    e = time.time()
    print("耗时", e - s)
