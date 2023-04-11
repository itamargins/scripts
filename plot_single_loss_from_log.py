import re
import matplotlib.pyplot as plt

# this script - 
# --- reads a full log file, filters only a single loss from it and plots over epoch

# PREP:
# --- specify the path of the log file
# --- specify the desired single loss to plot

###########################################################################################
# USER PARAMETERS

log_path = '/opt/imagry/CenterNet/src/exp/ddd_asu/default/logs_2023-02-06-13-58/'
single_loss = 'dep_entron_120_loss'


###########################################################################################

with open(log_path+'log.txt') as f:
    lines = f.readlines()
    
single_loss_data = []


for i in range(len(lines)):
	x = re.search(single_loss+" [0-9.]+", lines[i])
	single_loss_data.append(x.group().split(' '))
		
y = []
x = range(len(single_loss_data))
for i in x:
	y.append(float(single_loss_data[i][1]))

plt.title(single_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, y, marker = 'o', c = 'g')
plt.show()
