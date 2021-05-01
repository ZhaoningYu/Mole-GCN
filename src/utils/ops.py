import networkx as nx
from tqdm import tqdm
import math


class Generator(object):
    def __init__(self, data):
        self.data = data
        self.num_cliques = 0

    def gen_large_graph(self):
        g_list = self.data.g_list
        nodes_labels = self.data.node_labels
        cliques = self.gen_cliques(g_list)
        vocab = self.gen_vocab(cliques, nodes_labels)
        count_edges, count_nodes, cliques_weight, moles_weight = self.gen_edges(cliques, vocab, nodes_labels)
        g_graph = self.gen_final_graph(count_edges, count_nodes, cliques_weight, moles_weight, vocab)
        return g_graph, vocab

    def gen_cliques(self, g_list):
        cliques = []
        for g in tqdm(range(len(g_list)), desc="Gen_cliques", unit="graphs"):
            cliques.append([])
            # Find mcb
            mcb = self.find_minimum_cycle_basis(g_list[g])
            # convert elements of mcb to tuple
            mcb_tuple = [tuple(ele) for ele in mcb]

            # Find all edges not in cycles and add into cliques
            edges = []
            for e in g_list[g].edges:
                count = 0
                for c in mcb:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            cliques[g].extend(list(set(edges)))
            cliques[g].extend(mcb_tuple)
        return cliques

    def gen_vocab(self, cliques, nodes_labels):
        G_nodes = {}
        G_nodes_weights = {}
        G_vocab = {}
        for g in tqdm(range(len(cliques)), desc="Gen_vocab", unit="graphs"):
            # Generate G_nodes dictionary as vocabulary key tuple: (nodes, sum_of_edge_weights); value: 1 to number of cliques eg. ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)): 5
            # Using nodes label instead of nodes number
            for c in cliques[g]:
                le = []  # Nodes label lists
                lw = []  # Weights list of edges
                l_all = []
                if len(c) == 2:  # Bonds
                    for i in c:
                        le.append(nodes_labels[i - 1])
                    dic = self.data.g_list[g].get_edge_data(c[0], c[1])
                    lw.append(dic['weight'])
                else:  # Rings
                    for i in c:
                        le.append(nodes_labels[i - 1])
                    for i in range(len(c) - 1):
                        dic = self.data.g_list[g].get_edge_data(c[i], c[i + 1])
                        lw.append(dic['weight'])
                    dic = self.data.g_list[g].get_edge_data(c[-1], c[0])
                    lw.append(dic['weight'])
                global_cliques = tuple(le)
                l_all.append(global_cliques)
                if len(lw) == 1:
                    l_all.append(lw[0])
                else:
                    l_all.append(tuple(lw))
                l_all = tuple(l_all)
                if global_cliques not in G_nodes:
                    G_nodes[global_cliques] = len(G_nodes) + 1  # Value of G_nodes start from 1
                if l_all not in G_nodes_weights:
                    G_nodes_weights[l_all] = len(G_nodes_weights) + 1
        # Process G_nodes_weights
        keys = list(G_nodes_weights.keys())
        # Find the duplicated rings and bonds
        count_list = []  # values of all duplicated rings and bonds in dictionary
        for i in tqdm(range(len(keys)), desc="Find duplicated", unit="graph"):
            i_list = list(keys[i][0])
            for j in range(i + 1, len(keys)):
                inner_list = []
                if len(keys[i][0]) == 2:
                    inner_list.append(i_list)
                    inner_list.append(self.shift_right(i_list))
                    if list(keys[j][0]) in inner_list:
                        if keys[j][1] == keys[i][1]:
                            count_list.append(j)
                elif len(keys[i][0]) > 2:
                    i_inner_list = i_list
                    i_w_list = list(keys[i][1])
                    for t in range(len(i_list)):
                        if i_inner_list == list(keys[j][0]) and i_w_list == list(keys[j][1]):
                            count_list.append(j)
                        else:
                            i_inner_list = self.shift_right(i_inner_list)
                            i_w_list = self.shift_right(i_w_list)

        # Delete all duplicated rings and bonds
        g_nodes_weights = {key: val for key, val in G_nodes_weights.items() if val - 1 not in count_list}

        # Revalue the dictionary
        i = 1
        for key in g_nodes_weights.keys():
            G_vocab[key] = i
            i += 1

        # Extend G_vocab with all possible cliques sequences
        keys_list = list(G_vocab.keys())
        for v in tqdm(keys_list, desc="Extend vocab", unit='graphs'):
            lv = []
            lvw = []
            if len(v[0]) == 2:
                lv.append(v)
                if (tuple(self.shift_right(list(v[0]))), v[1]) not in lv:
                    lv.append((tuple(self.shift_right(list(v[0]))), v[1]))
                lv = tuple(lv)
                G_vocab[lv] = G_vocab[v]
                del G_vocab[v]
            elif len(v[0]) > 2:
                ring_list = []
                rings = v[0]
                weights = v[1]
                for i in range(len(v[0])):
                    lv.append(rings)
                    lvw.append(weights)
                    rings = tuple(self.shift_right(list(rings)))
                    weights = tuple(self.shift_right(list(weights)))
                for i in range(len(v[0])):
                    if (lv[i], lvw[i]) not in ring_list:
                        ring_list.append((lv[i], lvw[i]))
                ring_list = tuple(ring_list)
                G_vocab[ring_list] = G_vocab[v]
                del G_vocab[v]

        # Sort G_vocab
        sorted_g_vocab = {}
        sorted_key = sorted(G_vocab, key=G_vocab.get)
        for w in sorted_key:
            sorted_g_vocab[w] = G_vocab[w]
        return sorted_g_vocab

    def gen_edges(self, cliques, sorted_G_vocab, nodes_labels):
        # Count edges between cliques
        count_edges = []
        count_nodes = []
        all_nodes = []
        for g in tqdm(range(len(cliques)), desc="count node edge", unit="graph"):
            edges_dic = {}
            nodes_dic = {}
            clique_edges = []

            # Count all nodes
            for c in cliques[g]:
                x = list(c)
                index = self.find_index(x, self.data.g_list[g], sorted_G_vocab, nodes_labels)
                if index not in nodes_dic.keys():
                    nodes_dic[index] = 1
                else:
                    nodes_dic[index] += 1
                all_nodes.append(index)
            count_nodes.append(nodes_dic)
            # Find all edges between cliques
            for i in range(len(cliques[g])):
                for j in cliques[g][i]:
                    for t in range(i + 1, len(cliques[g])):
                        if j in cliques[g][t]:
                            clique_edges.append((cliques[g][i], cliques[g][t]))
            clique_edges = list(set(clique_edges))  # Delete duplicated edges

            # Count all edges
            for x, y in clique_edges:
                edge_keys = []
                index_x = self.find_index(x, self.data.g_list[g], sorted_G_vocab, nodes_labels)
                index_y = self.find_index(y, self.data.g_list[g], sorted_G_vocab, nodes_labels)
                edge_keys.append(index_x)
                edge_keys.append(index_y)
                edge_keys = sorted(edge_keys)
                if tuple(edge_keys) not in edges_dic.keys():
                    edges_dic[tuple(edge_keys)] = 1
                else:
                    edges_dic[tuple(edge_keys)] += 1
            count_edges.append(edges_dic)

        # Calculate edge weight
        edge_weight = {}
        for i in count_edges:
            for key in i.keys():
                if key not in edge_weight:
                    count = 0
                    for j in count_edges:
                        for keyj in j.keys():
                            if keyj == key:
                                count += 1
                    edge_weight[key] = count
        node_count = {}
        for i in count_nodes:
            for key in i.keys():
                if key not in node_count:
                    count = 0
                    for j in count_nodes:
                        for keyj in j.keys():
                            if key == keyj:
                                count += 1
                    node_count[key] = count
        cliques_weight = {}
        for i in count_edges:
            for key in i.keys():
                if key not in cliques_weight:
                    weight = math.log(
                        edge_weight[key] * len(count_edges) / (node_count[key[0]] * node_count[key[1]]))
                    if weight <= 0:
                        cliques_weight[key] = 0
                    else:
                        cliques_weight[key] = weight

        moles_weight = []
        mole_count = {}
        for i in count_nodes:
            for key in i.keys():
                if key not in mole_count:
                    mole_count[key] = 1
                else:
                    mole_count[key] += 1
        # print(mole_count)
        for i in count_nodes:
            m_weight = {}
            for key in i.keys():
                m_w = i[key] * math.log((len(count_nodes) + 1 / mole_count[key]) + 1)
                m_weight[key] = m_w
            moles_weight.append(m_weight)

        return count_edges, count_nodes, cliques_weight, moles_weight

    def gen_final_graph(self, count_edges, count_nodes, cliques_weight, moles_weight, sorted_G_vocab):
        # Generate the large graph
        c_nodes = []  # Cliques nodes
        for i in range(len(sorted_G_vocab)):
            c_nodes.append("C{}".format(i + 1))
        G_large = nx.Graph()
        G_large.add_nodes_from(c_nodes)
        m_nodes = []
        for i in range(len(self.data.g_list)):
            m_nodes.append("M{}".format(i + 1))
        G_large.add_nodes_from(m_nodes)

        # Add edges between cliques
        for i in count_edges:
            for key in i.keys():
                # If ignore the edge which weights are 0
                if cliques_weight[key] == 0:
                    continue
                else:
                    G_large.add_edge("C{}".format(key[0]), "C{}".format(key[1]), weight=cliques_weight[key])

        # Add edges between clique and mole
        for i in range(len(count_nodes)):
            for key in count_nodes[i].keys():
                G_large.add_edge("M{}".format(i + 1), "C{}".format(key), weight=moles_weight[i][key])
        return G_large

    def find_minimum_cycle_basis(self, g):
        return nx.cycle_basis(g)

    def shift_right(self, l):
        return [l[-1]] + l[:-1]

    def find_index(self, node, graph, vocab, nodes_labels):
        nodes_labels = nodes_labels
        index = -1
        x = list(node)
        g = graph
        sorted_G_vocab = vocab
        vocab_keys = list(sorted_G_vocab.keys())
        if len(x) == 2:
            w_inner = g.get_edge_data(x[0], x[1])['weight']
            for i in range(len(x)):
                x[i] = nodes_labels[x[i] - 1]
            for key in vocab_keys:
                if (tuple(x), w_inner) in key:
                    index = sorted_G_vocab[key]
        else:
            weight_list = []
            for i in range(len(x) - 1):
                e_label = g.get_edge_data(x[i], x[i + 1])
                weight_list.append(e_label['weight'])
            e_label = g.get_edge_data(x[-1], x[0])
            weight_list.append(e_label['weight'])
            for i in range(len(x)):
                x[i] = nodes_labels[x[i] - 1]
            for key in vocab_keys:
                if (tuple(x), tuple(weight_list)) in key:
                    index = sorted_G_vocab[key]
        return index