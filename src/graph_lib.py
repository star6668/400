import random
import math
import networkx as nx




def dag_generator(n, p, rng=None):
    """
    Fonction qui renvoie un graph orienté sans circuit (dag) avec pour chaque sommet une probabiltié p d'avoir une arête les reliant 
    """
    rng = rng or random
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G.add_edge(i, j)
    return G

def degeneracy(G):
    """
    G est le graph de co-comp d'un dag
    """
    core = nx.core_number(G)                
    return max(core.values(), default=0)    

def clique_number_exact(G):
    """
    G est le graph de co-comp d'un dag
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


def antichain_width(P: nx.DiGraph) -> int:
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
    matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=set(L))
    m_size = len(matching) // 2

    return n - m_size

def measure_cocomp_property(G, metric):
    """
    Mesure w(H_G) ou d(H_G) pour le DAG G.
    """
    if metric == "clique":
        return antichain_width(G)
    if metric == "degeneracy":
        return degeneracy(get_co_comp(G))
    raise ValueError('metric doit valoir "clique" ou "degeneracy".')



CLIQUE_MODEL = {
    "intercept": 1.948385941384962,
    "terms": [
        ("log_n", -0.6625087589448909),
        ("log_p", -0.3216239975347656),
        ("sqrt_p", -0.8646654660840488),
        ("log_n^2", -0.007323837993548654),
        ("log_n log_p", 0.04007553136270187),
        ("log_n sqrt_p", -0.02727836166017272),
        ("log_p^2", 0.101764332077841),
        ("log_p sqrt_p", 0.21092021576966044),
        ("sqrt_p^2", -0.5785280152187924),
        ("log_n^3", 0.0035557471610771097),
        ("log_n^2 log_p", 0.0020159010101700034),
        ("log_n^2 sqrt_p", -0.06026149331592526),
        ("log_n log_p^2", 0.008882714171391404),
        ("log_n log_p sqrt_p", -0.9697711498150182),
        ("log_n sqrt_p^2", -0.24201223030088648),
        ("log_p^3", -0.016083427098415454),
        ("log_p^2 sqrt_p", 0.06147038813651368),
        ("log_p sqrt_p^2", 5.0709859789531615),
        ("sqrt_p^3", -0.5155810367224651),
        ("log_n^4", 0.0001979173734670628),
        ("log_n^3 log_p", -0.0009961467389302053),
        ("log_n^3 sqrt_p", -0.008121504067277746),
        ("log_n^2 log_p^2", -0.006994620855008635),
        ("log_n^2 log_p sqrt_p", 0.045625175842130335),
        ("log_n^2 sqrt_p^2", 0.0835972783995482),
        ("log_n log_p^3", -0.009857438729332922),
        ("log_n log_p^2 sqrt_p", -0.22215860742573992),
        ("log_n log_p sqrt_p^2", 0.5585215733737254),
        ("log_n sqrt_p^3", 0.21713505812692008),
        ("log_p^4", -0.008367376328410784),
        ("log_p^3 sqrt_p", 0.025051768381266958),
        ("log_p^2 sqrt_p^2", -0.03723577138752322),
        ("log_p sqrt_p^3", -4.652900348602933),
        ("sqrt_p^4", -0.8999828702370415),
    ],
}


DEGENERACY_MODEL = {
    "intercept": -18.54393130910309,
    "terms": [
        ("log_n", -2.808983437617303),
        ("log_p", -0.35778675067172344),
        ("sqrt_p", -1.6139494087235673),
        ("log_n^2", 1.0032858583798732),
        ("log_n log_p", -0.24447736952782995),
        ("log_n sqrt_p", -6.153019126599521),
        ("log_p^2", -0.0267841698293633),
        ("log_p sqrt_p", 6.6576050858431035),
        ("sqrt_p^2", -0.02907046692824014),
        ("log_n^3", -0.19840665461576187),
        ("log_n^2 log_p", -0.14335549595212982),
        ("log_n^2 sqrt_p", 0.28223638994903916),
        ("log_n log_p^2", -0.09011916495932704),
        ("log_n log_p sqrt_p", 1.816122959282859),
        ("log_n sqrt_p^2", -8.028373370697166),
        ("log_p^3", 0.07728246976505715),
        ("log_p^2 sqrt_p", 0.9695459032896236),
        ("log_p sqrt_p^2", 27.83989664726159),
        ("sqrt_p^3", 1.6662223358452755),
        ("log_n^4", -0.00795150061928675),
        ("log_n^3 log_p", -0.10952008402316046),
        ("log_n^3 sqrt_p", 0.44518542314483506),
        ("log_n^2 log_p^2", -0.2691329506586916),
        ("log_n^2 log_p sqrt_p", 1.8100423984538652),
        ("log_n^2 sqrt_p^2", -4.05322181422922),
        ("log_n log_p^3", -0.26270872624811586),
        ("log_n log_p^2 sqrt_p", 2.2419145794440336),
        ("log_n log_p sqrt_p^2", -19.26181026094775),
        ("log_n sqrt_p^3", 29.632826973831325),
        ("log_p^4", -0.07863279621903092),
        ("log_p^3 sqrt_p", -1.0898642436042683),
        ("log_p^2 sqrt_p^2", 32.29655789069866),
        ("log_p sqrt_p^3", -64.31508740156623),
        ("sqrt_p^4", -6.770719774137262),
    ],
}


def feature_values(n, p):
    return {
        "log_n": math.log(n),
        "log_p": math.log(p),
        "sqrt_p": math.sqrt(p),
    }


def term_value(term, values):
    result = 1.0
    for factor in term.split(" "):
        if "^" in factor:
            name, power = factor.split("^")
            result *= values[name] ** int(power)
        else:
            result *= values[factor]
    return result


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def predict_from_model(model, n, p):
    values = feature_values(n, p)
    z = model["intercept"]
    for term, coef in model["terms"]:
        z += coef * term_value(term, values)
    return n * sigmoid(z)


def choose_p(target, metric, n):
    """
    Choisit p en resolvant la formule ajustee.

    metric doit valoir "clique" ou "degeneracy".
    """
    if metric != "clique" and metric != "degeneracy":
        raise ValueError('metric doit valoir "clique" ou "degeneracy".')

    if n <= 0:
        raise ValueError("n doit etre strictement positif.")
    if target <= 0:
        raise ValueError("target doit etre strictement positif.")

    lo = math.log(0.01)
    hi = math.log(0.9)

    for _ in range(80):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        p1 = math.exp(m1)
        p2 = math.exp(m2)
        if metric == "clique":
            e1 = abs(predict_from_model(CLIQUE_MODEL,n, p1) - target)
            e2 = abs(predict_from_model(CLIQUE_MODEL,n, p2) - target)
        else:
            e1 = abs(predict_from_model(DEGENERACY_MODEL,n, p1) - target)
            e2 = abs(predict_from_model(DEGENERACY_MODEL,n, p2) - target)
        if e1 <= e2:
            hi = m2
        else:
            lo = m1

    p = math.exp((lo + hi) / 2)
    if metric == "clique":
        return p, predict_from_model(CLIQUE_MODEL, n, p)
    else:
        return p, predict_from_model(DEGENERACY_MODEL, n, p)


def get_dag(
    n,
    *,
    clique_size=None,
    k_degen=None,
    max_attempts=1,
    tolerance=0,
    seed=None,
):
    """
    Genere un DAG simple en choisissant p avec les formules finales.

    Il faut fournir exactement une cible :
      - clique_size pour viser w(H_G) ;
      - k_degen pour viser d(H_G).

    tolerance est une erreur relative : 0.1 signifie 10%.
    """
    if (clique_size is None) == (k_degen is None):
        raise ValueError("Il faut fournir exactement une cible: clique_size ou k_degen.")
    if max_attempts <= 0:
        raise ValueError("max_attempts doit etre strictement positif.")
    if tolerance < 0:
        raise ValueError("tolerance doit etre positive.")

    metric = "clique" if clique_size is not None else "degeneracy"
    target = clique_size if clique_size is not None else k_degen
    p, predicted_value = choose_p(target, metric, n)
    rng = random.Random(seed) if seed is not None else random

    best_graph = None
    best_value = None
    best_error = None

    for attempt in range(1, max_attempts + 1):
        G = dag_generator(n, p, rng=rng)
        value = measure_cocomp_property(G, metric)
        error = abs(value - target) / target

        if best_error is None or error < best_error:
            best_graph = G
            best_value = value
            best_error = error

        if error <= tolerance or error < 0.001:
            break

    print(
        f"target : {target}\n"
        f"metric : {metric}\n"
        f"value : {best_value}\n"
        f"predicted_value : {predicted_value:.3f}\n"
        f"p : {p:.6f}\n"
        f"attempts : {attempt}"
    )
    return best_graph
