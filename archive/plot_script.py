import matplotlib.pyplot as plt
  
single_loss_log_path = '/opt/imagry/CenterNet/exp/ddd_asu/default/logs_2023-01-29-15-42/'  

x = []
y = []
for i,line in enumerate(open(single_loss_log_path+'one_loss_only.txt', 'r')):
#    print(i,line)
    lines = [i for i in line.split()]
    x.append(int(i))
#    print(x)
    y.append((float(lines[1])))
#    print(y)
      
plt.title(lines[0])
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.yticks(y)
plt.plot(x, y, marker = 'o', c = 'g')
  
plt.show()
