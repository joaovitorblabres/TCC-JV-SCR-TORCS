import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys

name = sys.argv[1]
col = int(sys.argv[2])

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open(name + '/eps.csv','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines[1:]:
        if len(line) > 1:
            x = line.split(',')
            xs.append(int(x[0]))
            ys.append(float(x[col]))
    ax1.clear()
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
