
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")

try:
    driver = GraphDatabase.driver(
        uri,
        auth=(username, password)
    )

    driver.verify_connectivity()
    print("Connected to Neo4j successfully!")

    with driver.session(database=database) as session:
        result = session.run("RETURN 'GraphRAG database is working!' AS message")
        print(result.single()["message"])

    driver.close()

except Exception as e:
    print("Neo4j connection failed:")
    print(e)
