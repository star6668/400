import networkx as nx
import random

def dag_generator(n, p):
    """
    Fonction qui renvoie un graph orienté sans circuit (dag) avec pour chaque sommet une probabiltié p d'avoir une arête les reliant 
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)
    return G

def degeneracy(G):
    """
    G est un dag
    """
    core = nx.core_number(G)                
    return max(core.values(), default=0)    

def clique_number_exact(G):
    """
    G est un dag
    """
    return max((len(C) for C in nx.find_cliques(G)), default=0)

def get_co_comp(G):
    """
    G est un dag
    renvoie le graph de co-comparabilité de G
    """
    TC = nx.transitive_closure_dag(G)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    nodes = list(G.nodes())
    nodes_size = len(nodes)
    for i in range(nodes_size):
        a = nodes[i]
        for j in range(i + 1, nodes_size):
            b = nodes[j]
            if not TC.has_edge(a, b) and not TC.has_edge(b, a):
                H.add_edge(a, b)
    return H


def max_clique_size_cocomp(G):
    """
    G est un dag
    Fonction optimisée pour les dag 
    taille max d'une clique du graphe de co-comparabilité de G.
    """
    TC = nx.algorithms.dag.transitive_closure_dag(G)
    n = TC.number_of_nodes()

    B = nx.Graph()
    L = [(v, "L") for v in TC.nodes()]
    Rr = [(v, "R") for v in TC.nodes()]
    B.add_nodes_from(L)
    B.add_nodes_from(Rr)

    for u, v in TC.edges():
        B.add_edge((u, "L"), (v, "R"))

    matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=set(L))
    nu = len(matching) // 2
    return n - nu

def max_parallelism_dilworth_descendants_nx(P: nx.DiGraph) -> int:
    """
    Dilworth (largeur = antichaîne max) via :
      1) adjacences u_L--v_R si v est atteignable depuis u (descendants)
      2) matching maximum biparti (NetworkX Hopcroft-Karp)
      3) retourne n - |M|
    """

    nodes = list(P.nodes())
    n = len(nodes)

    # Construire le biparti B
    L = [("L", u) for u in nodes]
    R = [("R", u) for u in nodes]

    B = nx.Graph()
    B.add_nodes_from(L, bipartite=0)
    B.add_nodes_from(R, bipartite=1)

    # Arêtes: u_L -- v_R si v est atteignable depuis u
    for u in nodes:
        for v in nx.descendants(P, u):
            B.add_edge(("L", u), ("R", v))

    # Matching maximum (renvoie un dict des deux côtés)
    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(B, top_nodes=set(L))
    m_size = len(matching) // 2

    return n - m_size
