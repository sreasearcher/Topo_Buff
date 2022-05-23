from off_util import *
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='SimHei', weight='bold')

offloading = Off()
gs=offloading.gs
ds = [
    [gs.nin_f,gs.nin_o,gs.nin_s],
    [gs.vgg_f,gs.vgg_o,gs.vgg_s],
    [gs.ale_f_final,gs.ale_o_final,gs.ale_s],
    [gs.res_f_final,gs.res_o_final,gs.res_s]
]

end = 15
lines = []
theo_ns=[]
min_times=[]
for i in range(len(ds)):
    line = []
    for j in range(1, end + 1):
        offloading.buf = j
        avg_time, _, _ = offloading.off_one(ds[i][0], ds[i][1], ds[i][2])
        line.append(avg_time)
    lines.append(line)
# print(lines)

x=list(range(1,end+1))


markevery=[]
for i in range(len(x)):
    if i%2:
        markevery.append(i)
l1,=plt.plot(x, lines[0], linestyle='--',
             marker='D', markevery=markevery)
l2,=plt.plot(x, lines[1], linestyle='-.',
             marker='*', markevery=markevery)
l3,=plt.plot(x, lines[2], linestyle=':',
             marker='+', markevery=markevery)
l4,=plt.plot(x, lines[3],
             marker='o', markevery=markevery)

fs = 19

plt.legend(handler_map = {l1:HandlerLine2D(numpoints=1)},
           handles=[l1, l2, l3, l4], labels=['NiN', 'VGG-16', 'AlexNet', 'ResNet-18'],
           loc='best', fontsize=fs)
plt.grid()
plt.xlabel('缓存大小', fontsize=fs)
plt.ylabel('平均任务时延（秒）', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.tight_layout()
plt.show()
