import pickle
import torch
import matplotlib.pyplot as plt

for epoch in range(40, 115):
    with open(f'./stored/ratioB_epoch{epoch}.pkl', 'rb') as f:
        ratioB = pickle.load(f)
    ratioB = torch.cat([r[r != 0].flatten() for r in ratioB])

    ratioB = ratioB.cpu().numpy()
    
    plt.hist(ratioB, bins=50, alpha=0.75, color='blue')
    plt.title('Distribution of RatioB')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    plt.savefig(f'./plot/ratios_epoch{epoch}.jpg', format='jpg', dpi=300)
    plt.close()
