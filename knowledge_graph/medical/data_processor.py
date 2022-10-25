import os
import csv
import json
from logger import Logger, tqdm

LOGGER = Logger("NEO4J")


class MedicalGraphDataProcessor:
    def __init__(self, input_path="medical.json", output_dir="import"):
        self.data_path = input_path
        self.output_dir = output_dir

        self.drugs = []  # 药品
        self.foods = []  # 食物
        self.checks = []  # 检查
        self.departments = []  # 科室
        self.producers = []  # 药品大类
        self.symptoms = []  # 症状

        self.disease_infos = []  # 疾病信息

        self.rels_department = []  # 科室－科室关系
        self.rels_noteat = []  # 疾病－忌吃食物关系
        self.rels_doeat = []  # 疾病－宜吃食物关系
        self.rels_recommandeat = []  # 疾病－推荐吃食物关系
        self.rels_commonddrug = []  # 疾病－通用药品关系
        self.rels_recommanddrug = []  # 疾病－热门药品关系
        self.rels_check = []  # 疾病－检查关系
        self.rels_drug_producer = []  # 厂商－药物关系

        self.rels_symptom = []  # 疾病症状关系
        self.rels_acompany = []  # 疾病并发关系
        self.rels_category = []  # 疾病与科室之间的关系

        self.read_examples()

    def read_examples(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for data in tqdm(f, desc=f">>> Reading data from {self.data_path}"):
                data_json = json.loads(data)
                disease = data_json['name']
                disease_dict = {'name': disease, 'desc': '', 'prevent': '', 'cause': '', 'easy_get': '',
                                'cure_department': '', 'cure_way': '', 'cure_lasttime': '', 'symptom': '',
                                'cured_prob': ''}

                if 'symptom' in data_json:
                    self.symptoms += data_json['symptom']
                    for symptom in data_json['symptom']:
                        self.rels_symptom.append([disease, symptom])

                if 'acompany' in data_json:
                    for acompany in data_json['acompany']:
                        self.rels_acompany.append([disease, acompany])

                if 'desc' in data_json:
                    disease_dict['desc'] = data_json['desc']

                if 'prevent' in data_json:
                    disease_dict['prevent'] = data_json['prevent']

                if 'cause' in data_json:
                    disease_dict['cause'] = data_json['cause']

                if 'get_prob' in data_json:
                    disease_dict['get_prob'] = data_json['get_prob']

                if 'easy_get' in data_json:
                    disease_dict['easy_get'] = data_json['easy_get']

                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                        self.rels_category.append([disease, cure_department[0]])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_department.append([small, big])
                        self.rels_category.append([disease, small])

                    disease_dict['cure_department'] = cure_department
                    self.departments += cure_department

                if 'cure_way' in data_json:
                    disease_dict['cure_way'] = data_json['cure_way']

                if 'cure_lasttime' in data_json:
                    disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                if 'cured_prob' in data_json:
                    disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_commonddrug.append([disease, drug])
                    self.drugs += common_drug

                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    for drug in recommand_drug:
                        self.rels_recommanddrug.append([disease, drug])

                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_noteat.append([disease, _not])

                    self.foods += not_eat
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_doeat.append([disease, _do])

                    self.foods += do_eat
                    recommand_eat = data_json['recommand_eat']

                    for _recommand in recommand_eat:
                        self.rels_recommandeat.append([disease, _recommand])
                    self.foods += recommand_eat

                if 'check' in data_json:
                    check = data_json['check']
                    for _check in check:
                        self.rels_check.append([disease, _check])
                    self.checks += check

                if 'drug_detail' in data_json:
                    drug_detail = data_json['drug_detail']
                    producer = [i.split('(')[0] for i in drug_detail]
                    self.rels_drug_producer += [[i.split('(')[0], i.split('(')[-1].replace(')', '')] for i in
                                                drug_detail]
                    self.producers += producer

                self.disease_infos.append(disease_dict)

    def write_nodes(self, label, nodes, ids):
        output_path = os.path.join(self.output_dir, f"{label}.csv")
        try:
            LOGGER.info(f">>> Creating {label} Nodes to {output_path}")
            with open(output_path, 'w+', encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{label}:ID", "name", ":LABEL"])
                for node_name, i in tqdm(zip(nodes, ids), desc=f">>> Creating {label} Nodes to Neo4j"):
                    writer.writerow([i, node_name, label])
        except Exception:
            LOGGER.error(">>> Failed to Create Nodes")

    def write_diseases_nodes(self, disease_infos, ids):
        output_path = os.path.join(self.output_dir, "Disease.csv")
        LOGGER.info(f">>> Creating Disease Nodes to {output_path}")
        with open(output_path, 'w+', encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Disease:ID", "name", "prevent", "cause", "desc", "easy_get",
                             "cure_lasttime", "cure_department", "cure_way", "cured_prob", ":LABEL"])
            for d, i in tqdm(zip(disease_infos, ids), desc=">>> Creating Disease Nodes to Neo4j"):
                writer.writerow([i, d["name"], d["prevent"], d["cause"], d["desc"], d["easy_get"],
                                 d["cure_lasttime"], d["cure_department"], d["cure_way"], d["cured_prob"], "Disease"])

    def write_graph_nodes(self):
        i = 0
        self.diseases_map = dict(zip([d["name"] for d in self.disease_infos], range(i, i + len(self.disease_infos))))
        self.write_diseases_nodes(self.disease_infos, range(i, i + len(self.disease_infos)))
        i += len(self.disease_infos)

        self.drugs = set(self.drugs)
        self.drugs_map = dict(zip(self.drugs, range(i, i + len(self.drugs))))
        self.write_nodes('Drug', self.drugs, range(i, i + len(self.drugs)))
        i += len(self.drugs)

        self.foods = set(self.foods)
        self.foods_map = dict(zip(self.foods, range(i, i + len(self.foods))))
        self.write_nodes('Food', self.foods, range(i, i + len(self.foods)))
        i += len(self.foods)

        self.checks = set(self.checks)
        self.checks_map = dict(zip(self.checks, range(i, i + len(self.checks))))
        self.write_nodes('Check', self.checks, range(i, i + len(self.checks)))
        i += len(self.checks)

        self.departments = set(self.departments)
        self.departments_map = dict(zip(self.departments, range(i, i + len(self.departments))))
        self.write_nodes('Department', self.departments, range(i, i + len(self.departments)))
        i += len(self.departments)

        self.producers = set(self.producers)
        self.producers_map = dict(zip(self.producers, range(i, i + len(self.producers))))
        self.write_nodes('Producer', self.producers, range(i, i + len(self.producers)))
        i += len(self.producers)

        self.symptoms = set(self.symptoms)
        self.symptoms_map = dict(zip(self.symptoms, range(i, i + len(self.symptoms))))
        self.write_nodes('Symptom', self.symptoms, range(i, i + len(self.symptoms)))

    def write_relationships(self, start_map, end_map, edges, rel_type, rel_name):
        # 去重处理
        set_edges = ['###'.join(edge) for edge in edges]
        for edge in tqdm(set(set_edges), desc=f">>> Creating {rel_type} Relations"):
            start_node, end_node = edge.split('###')
            try:
                start_id, end_id = start_map[start_node], end_map[end_node]
                self.writer.writerow([start_id, end_id, rel_name, rel_type])
            except Exception:
                print(edge)

    def write_graph_rels(self):
        output_path = os.path.join(self.output_dir, "relation.csv")
        LOGGER.info(f">>> Creating Relations to {output_path}")
        with open(output_path, 'w+', encoding="utf-8", newline="") as f:
            self.writer = csv.writer(f)
            self.writer.writerow([":START_ID", ":END_ID", "name", ":TYPE"])
            self.write_relationships(self.diseases_map, self.checks_map, self.rels_check, 'need_check', '诊断检查')
            self.write_relationships(self.diseases_map, self.foods_map, self.rels_recommandeat, 'recommand_eat', '推荐食谱')
            self.write_relationships(self.diseases_map, self.foods_map, self.rels_noteat, 'no_eat', '忌吃')
            self.write_relationships(self.diseases_map, self.foods_map, self.rels_doeat, 'do_eat', '宜吃')
            self.write_relationships(self.departments_map, self.departments_map, self.rels_department, 'belongs_to', '属于')
            self.write_relationships(self.diseases_map, self.drugs_map, self.rels_commonddrug, 'common_drug', '常用药品')
            self.write_relationships(self.producers_map, self.drugs_map, self.rels_drug_producer, 'drugs_of', '生产药品')
            self.write_relationships(self.diseases_map, self.drugs_map, self.rels_recommanddrug, 'recommand_drug', '好评药品')
            self.write_relationships(self.diseases_map, self.symptoms_map, self.rels_symptom, 'has_symptom', '症状')
            self.write_relationships(self.diseases_map, self.diseases_map, self.rels_acompany, 'acompany_with', '并发症')
            self.write_relationships(self.diseases_map, self.departments_map, self.rels_category, 'belongs_to', '所属科室')


if __name__ == '__main__':
    handler = MedicalGraphDataProcessor()
    handler.write_graph_nodes()
    handler.write_graph_rels()
