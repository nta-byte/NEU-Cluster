import torch as t
import numpy as np
import pickle
import os


class VAEDataset(t.utils.data.Dataset):
    def __init__(self, datain=None, path_datain=None):
        if path_datain:
            if os.path.exists(path_datain):
                with open(path_datain, 'rb') as f:
                    self.datain = pickle.load(f)
        else:
            self.datain = datain
        self.labels = self.datain['le'].transform(self.datain['labels'])

        # print(self.datain)

        '''results = {
            # 'filename': self.files,
            'features': self.output,
            'labels': self.labels,
            'le': self.le,
            'layer_name': 'fc1'
        }'''

    def __getitem__(self, index):
        vec = self.datain['features'][index]
        target = self.labels[index]
        # print (target)
        return vec, target

    def __len__(self):
        return len(self.datain['features'])

    # def get_classNum(self):
    #     return len(self.dataList[0])


# # %% Functions and classes -------------------
# class SingleCellDataset(t.utils.data.Dataset):
#     def __init__(self,
#                  sc_mat: pd.DataFrame,
#                  n_genes: int = -1):
#         self.sc_mat = sc_mat
#
#         self.sc_mat = self.sc_mat.iloc[:, (self.sc_mat.values == 0).sum(axis=0) > 0.05 * self.sc_mat.shape[0]]
#         self.sc_mat = self.sc_mat.iloc[self.sc_mat.values.sum(axis=1) > 100, :]
#
#         self.G = (n_genes if (n_genes > 0) & (n_genes < self.sc_mat.shape[1]) else self.sc_mat.shape[1])
#
#         self.Grt = int(np.sqrt(self.G))
#
#         self.cell_idx = self.sc_mat.index.tolist()
#         self.gene_names = self.sc_mat.columns.tolist()
#
#         self.sc_mat = self.sc_mat.values.astype(np.float32)
#         self.sc_mat = np.log(self.sc_mat + 1.0)
#         #        self.sc_mat = self.sc_mat / self.sc_mat.max(axis = 1).reshape(-1,1)
#
#         if self.G < self.sc_mat.shape[1]:
#             srtidx = np.argsort(self.sc_mat.sum(axis=0))[::-1]
#             self.sc_mat = self.sc_mat[:, srtidx[0:self.G]]
#             self.gene_names = [self.gene_names[x] for x in srtidx[0:self.G]]
#
#         self.data_len = len(self.cell_idx)
#
#     def getimage(self, index):
#         return self.gene_names, self.__getitem__(index)[0].numpy()
#
#     def __len__(self, ):
#         return self.data_len
#
#     def __getitem__(self, index):
#         name = self.cell_idx[index]
#         counts = t.tensor(self.sc_mat[index, :])
#         return counts, name


if __name__ == '__main__':
    brick_path = '/home/ntanh/ntanh/SDC_SALa_Clustering/CODE/NEU-Cluster/output_save_flow2/cifar10_RESNET18_pretrain_reduce_dim_UMAP_kmeanNinit50_pytorch_histeq_whitten/raw_features_train/RESNET18_pretrain_fc1_features_std_pytorch_histeq.pickle'
    datas = VAEDataset(path_datain=brick_path)
