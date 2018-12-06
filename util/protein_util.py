# author：chenhanping
# date 2018/11/30 下午2:13
# copyright ustc sse

# 和蛋白质处理有关的方法
import json
from Bio.PDB import *
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.DSSP import residue_max_acc
import os


def get_index_dict(structure) -> dict:
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
        if len(chain) < 20:
            continue
        for r in chain:
            if r.get_id()[0] == ' ' and is_aa(r):
                chain_dict[r.get_id()[1]] = i
                i = i + 1
        dict[chain.get_id()] = chain_dict
    return dict


def get_id_from_file(filename="pdb1aby"):
    if filename.startswith("pdb"):
        return filename[3:]
    else:
        return filename


class ProteinProcess:
    """
    蛋白质处理类，包括提取蛋白质序列，提取位点，生成res id和index对应关系等
    """
    def __init__(self, config_file_path="../config/path_config.json"):
        """
        初始化方法
        :param config_file_path: 配置文件的路径，配置所有与文件路径有关的信息
        """
        # 解析json配置文件
        with open(config_file_path, 'r') as f:
            paths = json.load(f)
        self.pdb_path = paths['pdb_path']
        self.site_json_path = paths['site_json_path']
        self.index_dict_path = paths['index_dict_path']
        self.protein_seq_path = paths['protein_seq_path']
        self.site_set_path = paths['site_path']
        self.p = PDBParser()
        self.ppb = PPBuilder()

    def get_structure(self, filename):
        """
        获取结构
        :param filename: 文件名,不需要加路径名
        :return:
        """
        id = get_id_from_file(filename)
        file_path = os.path.join(self.pdb_path, filename)
        structure = self.p.get_structure(id, file_path)
        return structure

    def save_index_dict(self, index_dict, id):
        for chain_id, d in index_dict.items():
            filename = id + "_" + chain_id + ".json"
            filepath = os.path.join(self.index_dict_path, filename)
            with open(filepath, 'w') as f:
                json.dump(d, f)
            print("保存", filename, "成功")

    def get_protein_seq(self, structure):
        """
            提取出复合物的所有肽链的序列
            :param structure:
            :return: dict
            """
        poly_seq_dict = {}
        model = structure[0]
        for chain in model:
            seq = ""
            for pp in self.ppb.build_peptides(chain):
                seq += str(pp.get_sequence())

            if len(seq) < 20:
                continue
            poly_seq_dict[chain.get_id()] = seq
        return poly_seq_dict

    def save_chain_seq(self, poly_seq_dict, id):
        """
        保存序列
        :param id: 蛋白质id
        :param poly_seq_dict:
        :return:
        """
        for chain_id, seq in poly_seq_dict.items():
            filename = id + "_" + chain_id + ".txt"
            filepath = os.path.join(self.protein_seq_path, filename)
            f = open(filepath, 'w')
            f.write(seq)
            print("保存", filename, "成功")

    def get_site_set(self, id, file_type="json")->dict:
        """
        获取位点信息
        :param id:
        :param file_type: 存储格式是itf或者json
        :return:
        """
        set_dict = {}
        if file_type == 'json':
            # 从json文件中读取位点
            with open(self.site_json_path + id + ".json", "r") as f:
                load_dict = json.load(f)
            for chain_id, dict_arr in load_dict.items():
                site_set = set()
                if len(dict_arr) == 0:
                    continue
                for item in dict_arr:
                    site_set.add(item['chain_index'])
                set_dict[chain_id] = site_set
        return set_dict

    def save_site_set(self, id, set_dict):
        """
        保存位点数据为txt
        :param id:
        :param set_dict:
        :return:
        """
        for chain_id, site_set in set_dict.items():
            filename = id + "_" + chain_id + ".txt"
            filepath = os.path.join(self.site_set_path, filename)
            # 写位点
            site_str = ','.join([str(x) for x in site_set])
            f = open(filepath, 'w')
            f.write(site_str)
            print("保存", filename, "成功")

    def process_data_from_dir(self):
        """
        将计算位点后的数据处理成文本
        :return:
        """
        json_list = os.listdir(self.site_json_path)
        for json in json_list:
            # 将json中的位点提取出来
            filename, _ = os.path.splitext(json)
            id = get_id_from_file(filename)
            pdb_file_name = "pdb" + id + ".ent"
            # 获取结构
            s = self.get_structure(pdb_file_name)
            site_set = self.get_site_set(id)

            self.save_site_set(id, site_set)
            # 获取蛋白质序列
            seq_dict = self.get_protein_seq(s)
            self.save_chain_seq(seq_dict, id)
            index_dict = get_index_dict(s)
            self.save_index_dict(index_dict, id)


if __name__ == '__main__':
    pp = ProteinProcess()
    pp.process_data_from_dir()
