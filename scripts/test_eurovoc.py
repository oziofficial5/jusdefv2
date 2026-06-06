import rdflib
from pathlib import Path

path = Path("data/eurovoc/eurovoc_in_skos_core_concepts.rdf")
print("Loading:", path)
g = rdflib.Graph()
g.parse(path.as_posix())
print("Triples:", len(g))