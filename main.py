# encoding=utf8
from randomwalk import decomposition
import networkx as nx
from operator import itemgetter
from BFS import sampleGraph

def getMembership(H):
    HT = np.transpose(H)
    M = HT / HT.sum(axis=1)[:, None] # rows are node probabilities, columns are communities
    return M
def getCommunities(M, clusters, alpha):
    detectedCommunities = []
    A = nx.adjacency_matrix(G).todense()
    nodes = sorted(nx.nodes(G))
    _, H = decomposition(A, clusters)
    M = getMembership(H)
    #print(np.transpose(np.round(M,2)))
    M[M >= alpha] = 1
    # we assume the probability of 0.75 is enough to include a node in a community
    M[M < 1] = 0
    #print(np.transpose(np.round(M,2)))
       
    # map indices of hyperedges in M to the corresponding nodes
    numbering = list(range(len(nodes)))  # create indices 
    for k in range(0, clusters):
        # extract indices of nodes with community membership = 1
        c_indices = [j for i, j in zip(M[:, k], numbering) if i == 1]
        if len(c_indices) == 1:
            c_nodes = [nodes[c_indices[0]]]
        else:
            # convert tuple to list
            c_nodes = list(itemgetter(*c_indices)(nodes))
        
        # Do a 1 step BSF to include the neighbours of the centroids
        temp = []
        for node in c_nodes:
            neighbours = sampleGraph(G, node, 1)
            if node not in temp:
                temp.append(node)
            temp.extend([int(v) for v in neighbours if v not in temp])
        temp = sorted(temp)
        detectedCommunities.append(temp)
    return detectedCommunities
    

    def save_embedding(self, emb_file, features):
        f_emb = open(emb_file, 'w')
        f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
        for i in range(len(features)):
            s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
            f_emb.write(s + "\n")
        f_emb.close()





def main():
    args = parse_args()

    model = BiasedWalk(args.graph, args.dimension, args.attributes, args.wt)
    features_matrix = \
        model.trans_mat(model.matrix0, model.matrix1, model.matrix2, model.matrix_conn, args.wt, args.alpha, args.step)
    model.save_embedding(args.output, features_matrix)


if __name__ == '__main__':
    main()
