import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

def make_corner(x,labels,label_names=None,axwidth=2):
    N = x.shape[1]
    blo,bhi = x.min(axis=0), x.max(axis=0)
    fig,axes = plt.subplots(N,N,figsize=(N*axwidth,N*axwidth))
    for i in range(N):
        for j in range(N):
            plt.sca(axes[i,j])
            plt.axis('off')

    unique_labels = sorted(list(set(labels)))
    patches = []
    for il,label in enumerate(unique_labels):
        mask = labels==label
        xlims = []
        for i in range(N):
            plt.sca(axes[i,i])
            plt.axis('on')
            h = plt.hist(x[mask,i],bins=20,density=True,histtype='step',color=f"C{il}")
            xlims.append(plt.gca().get_xlim())
        for i in range(1,N):
            for j in range(i):
                plt.sca(axes[i,j])
                plt.scatter(x[mask,j],x[mask,i],s=0.5,color=f"C{il}")
                plt.xlim(xlims[j])
        
        patches.append(Patch(label=label_names[label] if label_names is not None else label,color=f"C{il}"))
    
    plt.sca(axes[0,-1])
    plt.legend(handles=patches,ncol=3)
        