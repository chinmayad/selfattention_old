
import matplotlib.pyplot as plt
from matplotlib import gridspec


def visualize_maps(out,msg):
    n_ims=1
    n_feats=min(10,out[-1].shape[1])

    fig, axes = plt.subplots(len(out), n_feats*n_ims)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2048.0 / float(DPI), 1024.0 / float(DPI))


    for layer,ims in enumerate(out):
        cnt=0
        for i in range(n_ims):
            for j in range(n_feats):
                if j<ims.shape[1]:
                    #print(layer,j,cnt)
                    axes[layer, cnt].imshow(ims[i,j].cpu().data,cmap='gray')
                else:
                    axes[layer, cnt].set_visible(False)
                cnt=cnt+1
    plt.tight_layout()
    # axes[layer, cnt].set_title('im_%d,f_%d'%(i,j))
    #plt.show()
    plt.savefig(msg + '_maps')
    plt.clf()
    plt.close('all')


def visualize_all_maps(out,msg):
    im_idx=0
    channels=[out[i].shape[1] for i in range(len(out))]
    for i in range(len(channels)-1,0,-1):
        channels[i]=channels[i]-channels[i-1]

    n_layers=len(out)

    for layer in range(1,1+n_layers):
        cols=max(channels[:layer])

        fig = plt.figure(figsize=(cols + 1, layer + 1))

        gs = gridspec.GridSpec(layer, cols,
                               wspace=0.05, hspace=0.05,
                               top=1. - 0.5 / (layer + 1), bottom=0.5 / (layer + 1),
                               left=0.5 / (cols + 1), right=1 - 0.5 / (cols + 1))
        ims=out[layer-1]
        cnt=0

        for row in range(layer):
            for col in range(cols):
                if col < channels[row]:

                    ax = plt.subplot(gs[row, col])
                    ax.imshow(ims[im_idx, cnt].cpu().data, cmap='gray')
                    ax.set_axis_off()
                    cnt = cnt + 1

                else:
                    None#ax.set_visible(False)

                ax.set_xticklabels([])
                ax.set_yticklabels([])

        plt.savefig(msg + '_maps_layer%d'%(layer), bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close('all')


import torch.nn.functional as F
def visualize_classification(self, out,targets, msg):

    n_classes =self.linear.out_features
    n_inputs =self.linear.in_features

    fig, axes = plt.subplots(len(out), n_classes)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2048.0 / float(DPI), 1024.0 / float(DPI))

    target=targets[0]
    for layer, maps in enumerate(out):
        if maps.shape[1]==n_inputs:
            N,C,H,W=maps.shape
            maps=maps[0:1,:].detach().view(C,-1).permute(1,0)
            maps=self.linear(maps).permute(1,0).view(n_classes,H,W)
            maps=F.softmax(maps,dim=0)
            for c in range(n_classes):

                axes[layer, c].imshow(maps[c].cpu().data, cmap='gray',vmin=0,vmax=maps.max().item())
                if c == target:
                    axes[layer, c].set_title('*')
        else:
            for c in range(n_classes):
                axes[layer, c].set_visible(False)


    plt.tight_layout()
    plt.savefig(msg + '_classification')
    plt.clf()
    plt.close('all')
