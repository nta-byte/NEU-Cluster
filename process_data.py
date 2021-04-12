import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA(torch.nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.A = torch.nn.Linear(2, 3)
        self.B = torch.nn.Linear(3, 4)
        self.C = torch.nn.Linear(4, 4)
        self.D = torch.nn.Linear(4, 3)

    def forward(self, x):
        x = F.relu(self.A(x))
        x = F.relu(self.B(x))
        x = F.relu(self.C(x))
        x = F.relu(self.D(x))
        return x


class ModelB(torch.nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.A = torch.nn.Linear(2, 3)
        self.B = torch.nn.Linear(3, 4)
        self.C = torch.nn.Linear(4, 4)
        self.E = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.A(x))
        x = F.relu(self.B(x))
        x = F.relu(self.C(x))
        x = F.relu(self.E(x))
        return x


modelA = ModelB()
modelA_dict = modelA.state_dict()

print('-' * 40)
for key in sorted(modelA_dict.keys()):
    parameter = modelA_dict[key]
    print(key)
    # print(parameter.size())
    # print(parameter)


modelB = ModelB()
# modelB = nn.Sequential(*list(modelB.children()))
modelB.E = nn.Sequential()
modelB_dict = modelB.state_dict()

print('-' * 40)
for key in sorted(modelB_dict.keys()):
    parameter = modelB_dict[key]
    print(key)
    # print(parameter.size())
    # print(parameter)
#
# print('-' * 40)
# print("modelB is going to use the ABC layers parameters from modelA")
# pretrained_dict = modelA_dict
# model_dict = modelB_dict
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict)
# # 3. load the new state dict
# modelB.load_state_dict(model_dict)
# modelB_dict = modelB.state_dict()
# for key in sorted(modelB_dict.keys()):
#     parameter = modelB_dict[key]
#     print(key)
#     print(parameter.size())
#     print(parameter)
