import torch
import matplotlib.pyplot as plt

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.995)
lrs = []
ran = 100*30
for i in range(ran):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    #     print("Factor = ",0.1 if i!=0 and i%2!=0 else 1," , Learning Rate = ",optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(ran), lrs)
plt.savefig('1.jpg')
