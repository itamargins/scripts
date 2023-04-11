import re

log_path = '/opt/imagry/CenterNet/exp/ddd_asu/default/logs_2023-01-29-15-42/'

with open(log_path+'log.txt') as f:
    lines = f.readlines()
    

# TODO - generalize the searched loss to a string
single_loss = 'hm_loss'

with open(log_path+"one_loss_only.txt", "a") as f:
	f.truncate(0)
	#print(lines)
	for i in range(len(lines)):
		#x = re.search("min_dep [0-9.]+", lines[i])
		x = re.search(single_loss+" [0-9.]+", lines[i])
		#print(lines[i])
		#print(x.group())
		f.write(x.group() + "\n")
