from array import array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def heatmap(arr):
    plt.figure(1)
    plt.xlabel('image')
    plt.ylabel('text')
    plt.title('Visualization in a mini-batch')
    plt.imshow(arr)
    plt.colorbar()
    # plt.tight_layout()
    plt.savefig('output/visualize/heatmap.eps',dpi=500)
    plt.savefig('output/visualize/heatmap.jpg',dpi=500)
    

def tsne(image,text): 
    arr = np.vstack((text,image))
    tsne=TSNE(n_components=2).fit_transform(arr)
    plt.figure()
    plt.title('TSNE in a mini-batch')
    plt.scatter(tsne[:40,0],tsne[:40,1],c=np.arange(0,40))
    plt.scatter(tsne[3,0],tsne[3,1],c='r')
    plt.scatter(tsne[43,0],tsne[43,1],c='r')
    plt.text(np.mean(tsne[:,0]),np.mean(tsne[:,1]),'negative')
    plt.text(tsne[3,0],tsne[3,1],'positive')
    plt.text(tsne[43,0],tsne[43,1],'anchor')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/visualize/tsne.eps',dpi=500)
    plt.savefig('output/visualize/tsne.jpg',dpi=500)
    