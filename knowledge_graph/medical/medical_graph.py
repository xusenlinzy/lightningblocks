import json
import sys
from config import NEO4J_HOST, NEO4J_NAME, NEO4J_PASSWORD
from py2neo import Graph, Node
from logger import Logger, tqdm


LOGGER = Logger("NEO4J")


class MedicalGraph:
    def __init__(self, host=NEO4J_HOST, name=NEO4J_NAME,
                 password=NEO4J_PASSWORD, data_path="data/medical.json"):
        try:
            self.data_path = data_path
            self.graph = Graph(host=host, auth=(name, password))
            LOGGER.info(f">>> Successfully connect to Neo4j with IP: {NEO4J_HOST}")
            self.read_nodes()
        except Exception:
            LOGGER.error(">>> Failed to connect Neo4j")
            sys.exit(1)

    def clear(self):
        self.graph.delete_all()

    def reset(self):
        self.drugs = []  # 药品
        self.foods = []  # 食物
        self.checks = []  # 检查
        self.departments = []  # 科室
        self.producers = []  # 药品大类
        self.diseases = []  # 疾病
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

    def read_nodes(self):
        self.reset()
        with open(self.data_path, "r", encoding="utf-8") as f:
            for data in tqdm(f, desc=f">>> Reading data from {self.data_path}"):
                data_json = json.loads(data)
                disease = data_json['name']
                disease_dict = {'name': disease}
                self.diseases.append(disease)
                disease_dict['desc'] = ''
                disease_dict['prevent'] = ''
                disease_dict['cause'] = ''
                disease_dict['easy_get'] = ''
                disease_dict['cure_department'] = ''
                disease_dict['cure_way'] = ''
                disease_dict['cure_lasttime'] = ''
                disease_dict['symptom'] = ''
                disease_dict['cured_prob'] = ''

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
                    self.rels_drug_producer += [[i.split('(')[0], i.split('(')[-1].replace(')', '')] for i in drug_detail]
                    self.producers += producer
                    
                self.disease_infos.append(disease_dict)

    def create_nodes(self, label, nodes):
        try:
            LOGGER.info(f">>> Creating {label} Nodes to Neo4j")
            for node_name in tqdm(nodes, desc=f">>> Creating {label} Nodes to Neo4j"):
                node = Node(label, name=node_name)
                self.graph.create(node)
        except Exception:
            LOGGER.error(">>> Failed to Create Nodes to Neo4j")
            sys.exit(1)

    def create_diseases_nodes(self, disease_infos):
        LOGGER.info(">>> Creating Disease Nodes to Neo4j")
        for disease_dict in tqdm(disease_infos, desc=">>> Creating Disease Nodes to Neo4j"):
            node = Node("Disease",
                        name=disease_dict['name'],
                        desc=disease_dict['desc'],
                        prevent=disease_dict['prevent'],
                        cause=disease_dict['cause'],
                        easy_get=disease_dict['easy_get'],
                        cure_lasttime=disease_dict['cure_lasttime'],
                        cure_department=disease_dict['cure_department'],
                        cure_way=disease_dict['cure_way'],
                        cured_prob=disease_dict['cured_prob'])
            self.graph.create(node)

    def create_graph_nodes(self):
        self.create_diseases_nodes(self.disease_infos)
        self.create_nodes('Drug', set(self.drugs))
        self.create_nodes('Food', set(self.foods))
        self.create_nodes('Check', set(self.checks))
        self.create_nodes('Department', set(self.departments))
        self.create_nodes('Producer', set(self.producers))
        self.create_nodes('Symptom', set(self.symptoms))

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        # 去重处理
        set_edges = ['###'.join(edge) for edge in edges]
        LOGGER.info(f">>> Creating {rel_type} Relations to Neo4j")
        for edge in tqdm(set(set_edges), desc=f">>> Creating {rel_type} Relations to Neo4j"):
            edge = edge.split('###')
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, edge[0], edge[1], rel_type, rel_name)
            try:
                self.graph.run(query)
            except Exception:
                print(edge)

    def create_graph_rels(self):
        self.create_relationship('Disease', 'Check', self.rels_check, 'need_check', '诊断检查')
        self.create_relationship('Disease', 'Food', self.rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship('Disease', 'Food', self.rels_noteat, 'no_eat', '忌吃')
        self.create_relationship('Disease', 'Food', self.rels_doeat, 'do_eat', '宜吃')
        self.create_relationship('Department', 'Department', self.rels_department, 'belongs_to', '属于')
        self.create_relationship('Disease', 'Drug', self.rels_commonddrug, 'common_drug', '常用药品')
        self.create_relationship('Producer', 'Drug', self.rels_drug_producer, 'drugs_of', '生产药品')
        self.create_relationship('Disease', 'Drug', self.rels_recommanddrug, 'recommand_drug', '好评药品')
        self.create_relationship('Disease', 'Symptom', self.rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Disease', self.rels_acompany, 'acompany_with', '并发症')
        self.create_relationship('Disease', 'Department', self.rels_category, 'belongs_to', '所属科室')
    
    @staticmethod
    def write(file_path, data):
        with open(file_path, 'w+', encoding="utf-8") as f:
            f.write('\n'.join(list(set(data))))

    def export_data(self):
        self.write('dict/drug.txt', self.drugs)
        self.write('dict/food.txt', self.foods)
        self.write('dict/check.txt', self.checks)
        self.write('dict/department.txt', self.departments)
        self.write('dict/producer.txt', self.producers)
        self.write('dict/symptom.txt', self.symptoms)
        self.write('dict/disease.txt', self.diseases)


if __name__ == '__main__':
    handler = MedicalGraph()
    handler.export_data()

    handler.clear()
    handler.create_graph_nodes()
    handler.create_graph_rels()
