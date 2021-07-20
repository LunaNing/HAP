from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils_model import load

graph_embeddings = load('/home/ln/spyder-workspace/HAP-master-for-graph-classification/vis/PROTEINS/moa2-graph_emb_mat.pickle')
# graph_embeddings = load('/home/ln/graph_embedding.pickle')
graph_labels = load('/home/ln/spyder-workspace/HAP-master-for-graph-classification/vis/PROTEINS/moa2-graph_label_list.pickle')
print('graph embeddings:', graph_embeddings)
print('graph labels:', graph_labels)

tsne = TSNE(n_components=2, perplexity=10)
Y = tsne.fit_transform(graph_embeddings)
plt.figure()
plt.xticks([])
plt.yticks([])

plt.title('MOA')
plt.scatter(Y[:,0], Y[:,1], c=graph_labels)
plt.rcParams['figure.figsize'] = (20, 20)
plt.savefig('/home/ln/spyder-workspace/HAP-master-for-graph-classification/vis/PROTEINS/moa2-proteins1.pdf')
plt.show()

