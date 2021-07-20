import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import networkx as nx
import random
from utils_model import load


def visualize(orig_adj,targ_adj, att_mat, count):
    '''
		orig_adj: the adjacent matrix of an instance in graph list
		targetr_adj: the adjacent matrix of the instance after one coarsen layer
		att_mat: a list of att tensor between two nodes of two graphs
	'''
    path = '/home/ln/spyder-workspace/HAP-master-for-graph-classification/att_vis/IMDBMULTI'

    g_orig = nx.from_numpy_matrix(orig_adj)

    g_targ = nx.from_numpy_matrix(targ_adj)
    ag_targ = nx.nx_agraph.to_agraph(g_targ)


    row = att_mat.shape[0]
    column = att_mat.shape[1]

    g_union = nx.disjoint_union(g_orig, g_targ)
    ag_union = nx.nx_agraph.to_agraph(g_union)

    for i in range(row):
        for j in range(column):

            orig_idx = list(g_orig.nodes())[i]
            targ_idx = list(g_targ.nodes())[j]+len(g_orig.nodes())

            jet = plt.get_cmap('Reds')
            cNorm = colors.Normalize(vmin=0, vmax=att_mat.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            if att_mat[i][j] > 0:
                g_union.add_edge(orig_idx, targ_idx, weight=att_mat[i][j])

    nodelist_orig=[]
    nodelist_targ=[]
    edgelist_orig=[]
    edgelist_targ=[]
    edgelist_att=[]
    color_list=[]

    for i in list(g_union.nodes()):
        print('i:', i)
        if int(i)<len(list(g_orig.nodes())):
            nodelist_orig.append(i)
        elif int(i)>=len(list(g_orig.nodes())):
            nodelist_targ.append(i)

    for i in list(g_union.nodes()):
        for j in list(g_union.nodes()):
            if g_union.has_edge(i,j):
                if int(i) < len(list(g_orig.nodes())) and int(j)<len(list(g_orig.nodes())):
                    edgelist_orig.append((i,j))
                if int(i)>=len(list(g_orig.nodes())) and int(j)>=len(list(g_orig.nodes())):
                    edgelist_targ.append((i,j))
                if int(i) < len(list(g_orig.nodes())) and int(j)>=len(list(g_orig.nodes())):
                    edgelist_att.append((i,j))
                    color_list.append(ag_union.get_edge(j, j).attr['color'])


    pos = dict()
    pos.update((n, (random.uniform(1,8), random.uniform(1,20))) for i, n in enumerate(nodelist_orig))
    pos.update((n, (random.uniform(11,15), random.uniform(1,20))) for i, n in enumerate(nodelist_targ))
    nx.draw_networkx_nodes(g_union, pos=pos, nodelist=nodelist_orig, alpha=1,
                           node_color='tomato', node_size=20, linewidths=0.2)
    nx.draw_networkx_nodes(g_union, pos=pos, nodelist=nodelist_targ, alpha=1, node_color='tomato',
                            node_size=20, linewidths=0.2)
    nx.draw_networkx_edges(g_union, pos=pos, edgelist=edgelist_orig, width=0.2, edge_color='gray')
    nx.draw_networkx_edges(g_union, pos=pos, edgelist=edgelist_targ, width=0.2, edge_color='gray')
    nx.draw_networkx_edges(g_union, pos=pos,edgelist=edgelist_att, width=[float(d['weight']*2) for (u,v,d) in g_union.edges(data=True)], edge_color='mediumaquamarine')
    plt.axis('off')
    plt.savefig('/home/ln/spyder-workspace/HAP-master-for-graph-classification/att_vis/IMDBMULTI/example_{}.pdf'.format(count))


if __name__ == '__main__':
    count=6
    for i in range(1500):
        orig_adj = load('/home/ln/spyder-workspace/HAP-master-for-graph-classification/att_vis/IMDBMULTI/orig_adj_{}.pickle'.format(i))
        targ_adj = load('/home/ln/spyder-workspace/HAP-master-for-graph-classification/att_vis/IMDBMULTI/targ_adj_{}.pickle'.format(i))
        att_mat = load('/home/ln/spyder-workspace/HAP-master-for-graph-classification/att_vis/IMDBMULTI/att_mat_{}.pickle'.format(i))

        visualize(orig_adj[0], targ_adj[0], att_mat[0], count)
        count+=1

