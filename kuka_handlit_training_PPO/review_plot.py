import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def load_training():
    import os
    import re
    files=os.listdir("./trail1")
  
    saved_models_file_name = []
   
    regex = r"(reward_each_episode_)+(.*)+(.csv)"
    for f in files:
        match = re.search(regex,f)
        if match:
            saved_models_file_name.append(f)
        
    if len(saved_models_file_name) !=0:
        print("Privious trains are::")
        for i in range(0,len(saved_models_file_name)):
            print(str(i)+". "+saved_models_file_name[i])
        prompt = input("would you like to start from an old training?(y/n)")
        if prompt =="y":
            x=input("please enter which model would you like to load(number): ")

            filename = saved_models_file_name[int(x)]
            directory = "./trail1/"
            path = directory+filename
            return path
        else:
            return False
    else:
        return False

        
path = load_training()
def animate(i):
      
      env_name = "kuka_handlit-v0"
      nn_version =2.1
      rewardFunctionVersion=1
      graph_data = open(path,'r').read()
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
