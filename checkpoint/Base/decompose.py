import networkx as nx


def compute_top_level(G):
    tl = {}
    for node in nx.topological_sort(G):
        preds = [tl[v] + 1 for v in G.pred[node]]
        if preds:
            tl[node] = max(preds)
        else:
            tl[node] = 0
    return tl


def compute_subgraph_weight(subG):
    weight = 0
    for node in subG.nodes():
        weight += subG.nodes[node]['weight']
    return weight


def decompose_epigenomics(G):
    FJB = {}
    tl = compute_top_level(G)
    lmax = max(tl.values()) + 1
    level = [[None for _ in range(0)] for _ in range(lmax)]
    for node in tl.keys():
        l = tl[node]
        level[l].append(node)
    for i in range(lmax):
        level[i].sort()
    if lmax != 9:
        if len(level[0]) > 1:
            FJB[tuple(level[0])] = sum(level[1:], [])
        FJB[tuple(level[1])] = level[-1]
    else:
        FJB[tuple(level[0])] = sum(level[6:lmax], [])
        pal = sum(level[0:6], [])
        subG = G.subgraph(pal)
        components = nx.weakly_connected_components(subG)
        graphs = [G.subgraph(c) for c in components]
        for g in graphs:
            sFJB = decompose_epigenomics(g)
            FJB.update(sFJB)
    return FJB


def decompose_soykb(G):
    FJB = {}
    tl = compute_top_level(G)
    level = [[None for _ in range(0)] for _ in range(11)]
    for node in tl.keys():
        l = tl[node]
        level[l].append(node)
    for i in range(11):
        level[i].sort()
    FJB[tuple(level[0])] = sum(level[6:11], [])
    FJB[tuple(level[6])] = sum(level[7:11], [])
    for node in level[7]:
        if G.out_degree(node) == 0:
            level[7].remove(node)
    FJB[tuple(level[7])] = sum(level[8:11], [])
    FJB[tuple(level[8])] = sum(level[9:11], [])
    return FJB


def decompose_seismology(G):
    FJB = {}
    tl = compute_top_level(G)
    level = [[None for _ in range(0)] for _ in range(2)]
    for node in tl.keys():
        l = tl[node]
        level[l].append(node)
    for i in range(2):
        level[i].sort()
    FJB[tuple(level[0])] = level[1]
    return FJB


def decompose_srasearch(G):
    FJB = {}
    tl = compute_top_level(G)
    level = [[None for _ in range(0)] for _ in range(4)]
    for node in tl.keys():
        l = tl[node]
        level[l].append(node)
    for i in range(4):
        level[i].sort()
    for i in range(3):
        FJB[tuple(level[i])] = sum(level[i + 1:4], [])
    return FJB


def decompose_genome(G):
    gs = nx.number_weakly_connected_components(G)
    FJB = {}
    if (gs == 1):
        FJB = {}
        tl = compute_top_level(G)
        level = [[None for _ in range(0)] for _ in range(3)]
        for node in tl.keys():
            l = tl[node]
            level[l].append(node)
        for i in range(3):
            level[i].sort()
        for i in range(2):
            FJB[tuple(level[i])] = sum(level[i + 1:3], [])
        return FJB
    else:
        subGs = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
        for g in subGs:
            sFJB = decompose_genome(g)
            FJB.update(sFJB)
    return FJB


def decompose_cycle(G):
    gs = nx.number_weakly_connected_components(G)
    FJB = {}
    if (gs == 1):
        FJB = {}
        tl = compute_top_level(G)
        lmax = max(tl.values()) + 1
        level = [[None for _ in range(0)] for _ in range(lmax)]
        for node in tl.keys():
            l = tl[node]
            level[l].append(node)
        for i in range(lmax):
            level[i].sort()
        for i in range(lmax - 1):
            FJB[tuple(level[i])] = sum(level[i + 1:lmax], [])
        subs = G.subgraph(sum(level[lmax - 2:lmax], []))
        cps = nx.weakly_connected_components(subs)
        subG = [G.subgraph(c) for c in cps]
        for g in subG:
            l0 = [nd for nd in g.nodes if tl[nd] == lmax - 2]
            l1 = [nd for nd in g.nodes if tl[nd] == lmax - 1]
            l0.sort()
            l1.sort()
            FJB[tuple(l0)] = l1
        return FJB
    else:
        subGs = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
        for g in subGs:
            sFJB = decompose_cycle(g)
            FJB.update(sFJB)
    return FJB


def decompose_blast(G):
    FJB = {}
    tl = compute_top_level(G)
    lmax = max(tl.values()) + 1
    level = [[None for _ in range(0)] for _ in range(lmax)]
    for node in tl.keys():
        l = tl[node]
        level[l].append(node)
    for i in range(lmax):
        level[i].sort()
    for i in range(lmax - 1):
        FJB[tuple(level[i])] = sum(level[i + 1:lmax], [])
    return FJB


def decompose_bwa(G):
    FJB = {}
    tl = compute_top_level(G)
    lmax = max(tl.values()) + 1
    level = [[None for _ in range(0)] for _ in range(lmax)]
    for node in tl.keys():
        l = tl[node]
        level[l].append(node)
    for i in range(lmax):
        level[i].sort()
    for i in range(lmax - 1):
        FJB[tuple(level[i])] = sum(level[i + 1:lmax], [])
    return FJB


def decompose_montage(G):
    FJB = {}
    bottom = []
    top = []
    for node in G.nodes():
        if (G.out_degree(node) == 0):
            bottom.append(node)
        if (G.in_degree(node) == 0):
            top.append(node)
    top.sort()
    FJB[tuple(top)] = bottom
    n_bunch = [node for node in G.nodes() if node not in bottom]
    subsG = G.subgraph(n_bunch)
    cg = nx.weakly_connected_components(subsG)
    for g in cg:
        ng = subsG.subgraph(g)
        tl = compute_top_level(ng)
        lmax = max(tl.values()) + 1
        level = [[None for _ in range(0)] for _ in range(lmax)]
        for node in tl.keys():
            l = tl[node]
            level[l].append(node)
        for i in range(lmax):
            level[i].sort()
        for i in range(lmax - 1):
            FJB[tuple(level[i])] = sum(level[i + 1:lmax], [])
    return FJB
