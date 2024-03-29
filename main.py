import json
import os
import sys

with open('merged_file.json') as f:
    merge_data = json.load(f)

for value in merge_data:
    image_path = os.path.join('images', value['item']['slots'][0]['source_files'][0]['file_name'])
    
    print(image_path)

    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist")
        sys.exit(1)

