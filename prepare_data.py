# author：chenhanping
# date 2018/11/1 下午12:26
# copyright ustc sse

from Bio.PDB import PDBParser
from Bio.PDB import Select
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.DSSP import residue_max_acc
from SiteResidue import SiteResidue
import os


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
        for pp in ppb.build_peptides(chain):
            poly_seq_dict[chain.get_id()] = pp.get_sequence().tostring()
    return poly_seq_dict


def calculate_site(structure)->dict:
    """
    计算一个复合物中的结合位点
    :param structure:
    :return:
    """
    # 首先提取出结构中的所有多肽链，并写成pdb文件
    model = structure[0]
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
    # 计算所有的链的dssp
    for chain in model:
        # 记录每一个链中的位点
        site_list = list()
        chain_file_name = "chain/"+structure.get_id()+"_"+chain.get_id()+".ent"
        chain_dict = dssp_dict_from_pdb_file(chain_file_name, DSSP="./mkdssp")[0]
        # dict的形式如
        # ('A', (' ', 1, ' ')): ('M', '-', 108, 360.0, -92.4, 1, 0, 0.0, 2, -0.4, 0, 0.0, 127, -0.1)
        # step2 ,遍历chain，判断残基是否占最大可达面积的16%
        for item in chain_dict.items():
            residue_index = item[0][1][1]
            residue = chain[residue_index]
            # 获取这个残基的acc
            acc = float(item[1][2])
            # 获取这个残基可以达到的最大acc
            max_acc = max_acc_dict[residue.get_resname()]
            if acc / max_acc > 0.16:
                # step3, 比较当前的acc和在复合物中的acc的差值是否大于1
                # 获取复合物中的当前残基的acc, item[0]是复合物中dict的key，也就是residue的唯一标识
                # 获取到的dict形式与上述的dict形式一致，所以结果元组的第三位就是acc
                complex_acc = complex_dssp_dict[item[0]][2]
                if acc - complex_acc > 1:
                    # 这就是位点了
                    site_residue = SiteResidue()
                    site_residue.acc = acc
                    site_residue.chain_id = chain.get_id()
                    site_residue.chain_index = residue_index
                    site_residue.seq = poly_seq_dict[chain.get_id()][residue_index-1]
                    site_list.append(site_residue)
        site_dict[chain.get_id()] = site_list
    return site_dict


def start():
    structure_id = "1aby"
    filename = "data/1aby.pdb"
    p = PDBParser()
    structure = p.get_structure(structure_id, filename)
    # 处理蛋复合物
    process_complex(structure)
    clean_file_name = "clean_data/1aby.ent"
    structure = p.get_structure(structure_id, clean_file_name)
    dict = calculate_site(structure)
    for item in dict.items():
        print(item[1][0], len(item[1]))


if __name__ == '__main__':
    start()