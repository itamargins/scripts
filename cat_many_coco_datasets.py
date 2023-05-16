import os
import json
import argparse
import glob

parser = argparse.ArgumentParser(description='cat coco format json files')

parser.add_argument('-j', '--json_files', nargs='*', required=True,
                            help='sapace sepearated json files in coco format.')
parser.add_argument("-o", '--output_path', help = 'new json output path.',
                    default = ".", type = str)
parser.add_argument("-t", '--tag', help = 'json file basename.',
                    default = "joined_data", type = str)
args = parser.parse_args()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

class Data:
    def __init__(self):
        self.data = {'images': [], 'annotations': [], 'categories': []}
        self.image_counter = 0
        self.annotation_counter = 0
        self.sim_set = set()
        # TODO - do annotations start from index 1? do images start from 0?

    def add_data(self,all_data):
        # for each simulation in the set to add
        for index,new_data in enumerate(all_data):
            print(f'CATTING: {index+1} / {len(all_data)}')
            self.image_index_mapping = {}
            # self.annotation_idx_mapping = {}
            # TODO - is there a need for annotation_idx_mapping?

            self.add_images(new_data)
            self.add_annotations(new_data)
            if index == 0:
                self.data['categories'] = new_data['categories']
            else:
                assert(self.data['categories'] == new_data['categories'], 
                       '"CATEGORIES" ENTRIES ARE NOT THE SAME')
        print("FINISHED")

    def add_images(self, new_data):
        for index,image_data in enumerate(new_data['images']):
            # TODO - check if the new image is already in our accumulated data, so we avoid duplicates
            self.image_counter += 1
            self.sim_set.add(image_data['file_name'].split('/')[-3])
            transformed_image_data = self.transform_image_data(image_data)
            self.data['images'].append(transformed_image_data)

    def add_annotations(self, new_data):
        for index,ann_data in enumerate(new_data['annotations']):
            self.annotation_counter += 1
            transformed_ann_data = self.transform_ann_data(ann_data)
            self.data['annotations'].append(transformed_ann_data)

    def transform_image_data(self, image_data):
        transformed_image_data = image_data.copy()
        self.image_index_mapping.update({image_data['id'] : self.image_counter})
        transformed_image_data['id'] = self.image_counter
        return transformed_image_data
    
    def transform_ann_data(self, ann_data):
        transformed_ann_data = ann_data.copy()
        transformed_ann_data['image_id'] = self.image_index_mapping[ann_data['image_id']]
        transformed_ann_data['id'] = self.annotation_counter
        return transformed_ann_data

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
def read_all_data(json_files):
    all_data = []
    for index,json_file in enumerate(json_files):
        print(f'READING: {json_file}, {index+1} / {len(json_files)}')
        with open(json_file,'r') as json_file:
            all_data.append(json.load(json_file))
    return all_data

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# json_files = glob.glob('/home/imagry/DepthData/Depth_annotation_data/cognata/20230514_102921_itamar/**/coco_converted_gt.json',recursive=True)
json_files = glob.glob('/home/imagry/DepthData/Depth_annotation_data/cognata/*.json',recursive=True)
print(len(json_files))
# json_files = json_files[:5]
args.output_path = os.getcwd()
args.tag = "cognata_490sim"
json_output_path = os.path.join(args.output_path, args.tag + '.json')

all_data = read_all_data(json_files)
output_data = Data()
output_data.add_data(all_data)

print(f'TOTAL NUMBER OF SIMULATIONS: {len(output_data.sim_set)}')
print(f'TOTAL NUMBER OF IMAGES: {output_data.image_counter}')
print(f'TOTAL NUMBER OF OBJECTS: {output_data.annotation_counter}')

json_object = json.dumps(output_data.data, indent=4)
with open(json_output_path, "w") as outfile:
    outfile.write(json_object)