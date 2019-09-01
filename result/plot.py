import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
      env_name = "kuka_handlit-v0"
      nn_version =3
      rewardFunctionVersion=1
      graph_data = open("4.csv".format(env_name,rewardFunctionVersion,nn_version),'r').read()
      lines = graph_data.split('\n')
      xs = []
      ys = []
      for line in lines:
            if len(line)>1:
                  x,y = line.split(',')
                  xs.append(round(float(x)))
                  ys.append(round(float(y)))
      ax1.clear()
      ax1.plot(ys, xs)

ani = animation.FuncAnimation(fig, animate, interval = 1000)
plt.show()
