import networkx as nx


THRESHOLD=0.01
def amdahlM(size, alpha, proc):
    runtime = 0.0
    for i in range(len(size)):
        runtime += alpha[i] * size[i] + (1 - alpha[i]) * size[i] / proc
    return runtime


def amdahl(size, alpha, proc):
    return alpha * size + (1 - alpha) * size / proc


def compute_benefit(pj_ko, pj_k, ko, k):
    return pj_ko / ko - pj_k / k


def compute_rel_runtime_improvement(pj_ko, pj_k):
    return (pj_ko - pj_k) / pj_ko


def compute_W(G, proc):
    workload = 0.0
    for node in G.nodes:
        workload += G.nodes[node]['d_weight'] * proc[node]
    return workload


def compute_L(G):
    bl, cp = compute_bottom_weight(G)
    root_list = []
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            root_list.append(node)
    root = root_list[0]
    for node in root_list:
        if (bl[node]) > bl[root]:
            root = node
    cp_task = []
    cp_task.append(root)
    while len(G.succ[root]) > 0:
        max_child = None
        for node in G.succ[root]:
            if max_child is None:
                max_child = node
            else:
                if bl[node] > bl[max_child]:
                    max_child = node
        cp_task.append(max_child)
        root = max_child
    return cp, cp_task



def compute_pred_level(G):  # The precedence level denotes the shortest path to a node from the source node
    pl = {}
    max_level = 0
    for node in nx.topological_sort(G):
        preds = [pl[v] + 1 for v in G.pred[node]]
        if preds:
            pl[node] = min(preds)
        else:
            pl[node] = 0
        max_level = max(max_level, pl[node])
    return pl, max_level


def compute_bottom_weight(G):
    bl = {}
    cp = 0
    chain = list(nx.topological_sort(G))
    for node in reversed(chain):
        succs = [bl[v] + G.nodes[node]['d_weight'] for v in G.succ[node]]
        if succs:
            bl[node] = max(succs)
        else:
            bl[node] = G.nodes[node]['d_weight']
        cp = max(cp, bl[node])
    return bl, cp


def computeMd(tl, visited, assign):
    md = [0] * (max(tl.values()) + 1)
    for node in visited:
        if visited[node] == 1:
            md[tl[node]] += assign[node]
    return md


def getParentsNb(G):
    par = {}
    for node in G.nodes:
        par[node] = len(list(G.predecessors(node)))
    return par


def C(i, j, L, G):
    ck = 0
    f_list = {}
    for task in L[i:j]:
        ck += G.nodes[task]['ck_ext']
        for node in G.successors(task):
            if node not in L[i:j]:
                f_list.update(G.edges[task, node]['filelist'])
    ck+=sum(f_list.values())
    return ck

def R(i,j,L,G):
    f_list={}
    for task in L[i:j]:
        for node in G.predecessors(task):
            if node not in L[i:j]:
                f_list.update(G.edges[node,task]['filelist'])
    return sum(f_list.values())

def T(i, j, L, G):
    weight = 0
    rv=0
    f_list={}
    for task in L[i:j]:
        weight += G.nodes[task]['d_weight']
        for node in G.predecessors(task):
            if node not in L[i:j]:
                f_list.update(G.edges[node,task]['filelist'])
    for value in f_list.values():
        rv += value
    return weight,rv

def compute_average(G):
    sum = 0.0
    nb = nx.number_of_nodes(G)
    for node in G.nodes:
        sum += G.nodes[node]['weight']
    return sum / nb