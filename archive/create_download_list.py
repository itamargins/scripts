import json
import os
import random

#json_path = "/home/imagry/offline_data/2023-01-05_full_model_2619bfb_exp/192_3d_asu_train.json"
path = "/home/imagry/offline_data/first_train_data/"
json_name = "192_3d_asu_train.json"
download_command_file_name = "download_commands.sh"

json_path = path + json_name
image_target_path = path + "images/"

with open(json_path, 'r') as f:
	data = json.load(f)

print(len(data['images']))
print(data['images'][0])

#image_idx_range = list(range(550001,550002))
image_idx_range = random.sample(range(1,10),3)

short_list = [data['images'][ix] for ix in image_idx_range]
print(short_list)

new_data = {"images":[], "annotations":[]}

print(new_data)

with open((path + download_command_file_name), 'w+') as f:
	base_str = 'sshpass -p "q9rHOJOnco/o" scp imagry@192.168.1.89:XYZ ' + image_target_path
	for ix, img_info in enumerate(short_list):
#		print(f'ix: {ix},\n img_info: {img_info}')
#		print(f'base_str BEFORE: {base_str}')
		new_name = base_str.replace("XYZ", img_info['file_name']) + "\n"
#		print(f'new_name: {new_name}')
		new_path = os.path.join(path, os.path.basename(img_info['file_name'].strip())+ image_target_path)
		
#		data['images'][ix]['file_name'] = new_path
#		data['images'][0:1][ix]['file_name'] = new_path
#		print(new_path)
#		f.write(new_name)

		new_data['images'].append(img_info)
		new_data['images'][ix]['file_name'] = new_path

	print(f'new_data: {new_data}')
	new_data["annotations"] = data["annotations"]
	print(f'new_data: {new_data}')
	#data['images'] = data['images'][0:10]


	new_data_object = json.dumps(new_data, indent=4)
	f.write(new_data_object)
#	json.dump(new_data, json_path)

