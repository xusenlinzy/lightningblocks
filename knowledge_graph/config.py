import os

############## Neo4j Configuration ###############
NEO4J_HOST = os.environ.get('NEO4J_HOST', '192.168.0.55')
NEO4J_NAME = os.getenv("NEO4J_NAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "123456")
