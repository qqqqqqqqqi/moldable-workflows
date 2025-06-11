import json
import random
import uuid
import networkx as nx



# WfGen
def build_workflow(filename):
    filename = f"../WfGen/{filename}.json"
    with open(filename, 'r') as f:
        json_data = f.read()
    data = json.loads(json_data)
    G = nx.DiGraph()
    dep = {}
    file_weight = {}
    comm=0
    comp=0
    for t in data['workflow']['tasks']:
        dep[t['name']] = [[], []]
        ck = 0.0
        rv = 0.0
        for c in t['files']:
            if (c['link'] == 'input'):
                rv += c['sizeInBytes']
                dep[t['name']][1].append(c['name'])
            elif c['link'] == 'output':
                ck += c['sizeInBytes']
                dep[t['name']][0].append(c['name'])
            file_weight[c['name']] = c['sizeInBytes']
        comm+=ck
        comp+=t['runtimeInSeconds']
        G.add_node(t['name'], weight=t['runtimeInSeconds'], ck=ck, rv=rv, ck_ext=0, alpha=random.uniform(0, 0.25))

    for t in data['workflow']['tasks']:
        if len(t['children']) == 0:
            # exit output
            G.nodes[t['name']]['ck_ext'] = sum([file_weight[f] for f in dep[t['name']][0]])
        else:
            for c in t['children']:
                w = {}
                temp = 0
                for f in dep[t['name']][0]:  # output of father
                    if f in dep[c][1]:  # input of child
                        w[f] = file_weight[f]
                        temp += file_weight[f]
                G.add_edge(t['name'], c, filelist=w, temp=temp)
    return G,comm/comp


# daggen
def build_random_dag(filename,plat):
    tG = nx.nx_agraph.read_dot(f'./DAGGEN/{filename}')
    G = nx.DiGraph()
    comm=0
    comp=0
    for node in tG.nodes:
        comp += int(tG.nodes[node]['size'])/plat.flops
        G.add_node(node, weight=float(tG.nodes[node]['size']) / plat.flops, alpha=float(tG.nodes[node]['alpha']))
        ck = 0
        succs = list(tG.successors(node))
        if succs:
            transfer = float(tG.edges[node, succs[0], 0]['size'])
            comm += transfer
            ck = transfer
            f = str(uuid.uuid4())
            for succ in succs:
                G.add_edge(node, succ, filelist={f: transfer}, temp=transfer)
        G.nodes[node]['ck'] = ck
        G.nodes[node]['ck_ext'] = 0
    return G,comm/comp

