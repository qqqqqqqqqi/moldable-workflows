import networkx as nx

THRESHOLD = 0.01


def amdahlM(size, alpha, proc):
    runtime = 0.0
    for i in range(len(size)):
        runtime += alpha[i] * size[i] + (1 - alpha[i]) * size[i] / proc
    return runtime


def amdahl(size, alpha, proc):
    return alpha * size + (1 - alpha) * size / proc


def rooflineM(size, max_degree, proc):
    runtime = 0.0
    for i in range(len(size)):
        runtime += size[i] / min(max_degree[i], proc)
    return runtime


def roofline(size, max_degree, proc):
    return size / min(max_degree, proc)


def communicationM(size, cc, proc):
    runtime = 0.0
    for i in range(len(size)):
        runtime += size[i] / proc + cc[i] * (proc - 1)
    return runtime


def communication(size, cc, proc):
    return size / proc + cc * (proc - 1)


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


def compute_top_level(G):
    tl = {}
    max_level = 0
    for node in nx.topological_sort(G):
        preds = [tl[v] + 1 for v in G.pred[node]]
        if preds:
            tl[node] = max(preds)
        else:
            tl[node] = 0
        max_level = max(max_level, tl[node])
    return tl, max_level


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


def compute_top_weight(G):
    tl = {}
    for node in nx.topological_sort(G):
        preds = [tl[v] + G.nodes[v]['d_weight'] for v in G.pred[node]]
        if preds:
            tl[node] = max(preds)
        else:
            tl[node] = 0
    return tl


def compute_bottom_level(G):
    bl = {}
    chain = list(nx.topological_sort(G))
    for node in reversed(chain):
        succs = [bl[v] + 1 for v in G.succ[node]]
        if succs:
            bl[node] = max(succs)
        else:
            bl[node] = 0
    return bl


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
    for task in L[i:j]:  # (i+1,j)
        ck += G.nodes[task]['ck_ext']
        for node in G.successors(task):
            if node not in L[i:j]:
                f_list.update(G.edges[task, node]['filelist'])
    ck += sum(f_list.values())
    return ck


def R(i, j, L, G):
    f_list = {}
    for task in L[i:j]:
        for node in G.predecessors(task):
            if node not in L[i:j]:
                f_list.update(G.edges[node, task]['filelist'])
    return sum(f_list.values())


def T(i, j, L, G):
    weight = 0
    rv = 0
    f_list = {}
    for task in L[i:j]:
        weight += G.nodes[task]['d_weight']
        for node in G.predecessors(task):
            if node not in L[i:j]:
                f_list.update(G.edges[node, task]['filelist'])
    for value in f_list.values():
        rv += value
    return weight, rv


def precompute_successors(G):
    succ_dict = {}
    for node in G.nodes():
        succ_dict[node] = list(G.successors(node))
    return succ_dict


def precompute_predecessors(G):
    pred_dict = {}
    for node in G.nodes():
        pred_dict[node] = list(G.predecessors(node))
    return pred_dict


def pre_compute_topology(L, G):
    n = len(L)
    prefix_w = [0] * (n + 1)
    ck_exit = [0] * n
    m_r = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    m_c = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n):
        prefix_w[i + 1] = prefix_w[i] + G.nodes[L[i]]['d_weight']
        ck_exit[i] = G.nodes[L[i]]['ck_ext']
        for j in range(i + 1, n + 1):
            for pred in G.predecessors(L[j - 1]):
                if pred not in L[i:j]:
                    m_r[i][j].update(G.edges[pred, L[j - 1]]['filelist'])
            for succ in G.successors(L[i]):
                if succ not in L[i:j]:
                    m_c[i][j].update(G.edges[L[i], succ]['filelist'])
    prefix_r = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    prefix_c = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    for j in range(n, 0, -1):
        for i in range(j - 1, -1, -1):
            prefix_c[i][j].update(prefix_c[i + 1][j])
            prefix_c[i][j].update(m_c[i][j])
    for i in range(0, n):
        for j in range(i + 1, n + 1):
            prefix_r[i][j].update(prefix_r[i][j - 1])
            prefix_r[i][j].update(m_r[i][j])
    return prefix_w, prefix_r, prefix_c, ck_exit


def pre_compute_RTC(L, G):
    n = len(L)
    prefix_w = [0] * (n + 1)
    rv = {}
    ck = {}
    for i, task in enumerate(L):
        prefix_w[i + 1] = prefix_w[i] + G.nodes[task]['d_weight']
        ck[task] = G.nodes[task]['ck']
        rv[task] = G.nodes[task]['rv']
    return prefix_w, ck, rv


def compute_average(G):
    sum = 0.0
    nb = nx.number_of_nodes(G)
    for node in G.nodes:
        sum += G.nodes[node]['weight']
    return sum / nb


def compute_max_parallel(G):
    layers = list(nx.topological_generations(G))
    return max([len(layer) for layer in layers])


def compute_longest_weighted_path(G, weight):
    dist = {}
    for node in nx.topological_sort(G):
        time_succ = weight[node]
        preds = [dist[v] + time_succ for v in G.pred[node]]
        if preds:
            dist[node] = max(preds)
        else:
            dist[node] = time_succ
    return max(dist.values())
