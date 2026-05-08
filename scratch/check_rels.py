from backend.services.graph import GraphService
import os

def check_rels():
    try:
        gs = GraphService()
        gs.connect()
        with gs.driver.session() as session:
            res = session.run('MATCH ()-[r]->() RETURN type(r) as type, count(r) as count')
            results = [(record['type'], record['count']) for record in res]
            with open('rel_check.txt', 'w') as f:
                f.write(str(results))
    except Exception as e:
        with open('rel_check.txt', 'w') as f:
            f.write(f"Error: {str(e)}")

if __name__ == "__main__":
    check_rels()
