import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

with open('./watchlist.json') as data_file:
    data = json.load(data_file)

y1 = data['test']['mlogloss']
x1 = [i for i in range(1, len(y1)+1)]

y2 = data['train']['mlogloss']

plt.plot(x1, y1, label='test')
plt.plot(x1, y2, label='train')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
ybox1 = TextArea("", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
ybox2 = TextArea("",     textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
ybox3 = TextArea("", textprops=dict(color="b", size=15,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=5)

ax = plt.subplot(111)

anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.08, 0.4),
                                  bbox_transform=ax.transAxes, borderpad=0.)

ax.add_artist(anchored_ybox)
plt.legend()
plt.show()
