import argparse

# REPLCE STRING IN JSON FILE

parser = argparse.ArgumentParser(description='replace string in json')
parser.add_argument('--json_in', '-i',
                    dest = 'json_in', 
                    type = str,
                    help = 'input json'
                    )

parser.add_argument('--json_out', '-o',
                    dest = 'json_out', 
                    type = str,
                    help = 'output json',
                    required=False
                    )

parser.add_argument('--replace', '-r',
                    dest = 'replace', 
                    type = str,
                    help = 'string to find and replace',
                    required=True
                    )

parser.add_argument('--with', '-w',
                    dest = 'replacement', 
                    type = str,
                    help = 'string to replace instead of the original',
                    required=True
                    )
args = parser.parse_args()

JSON_IN = args.json_in
JSON_OUT = args.json_out if args.json_out is not None else args.json_in
findString = args.replace
replaceString = args.replacement
  
print(f'REPLACING: "{findString}" ---> "{replaceString}" ')
with open(JSON_IN, 'r') as f:
    data = f.read()
    data = data.replace(findString, replaceString)
  
print(f'WRITING TO: "{JSON_OUT}"')
with open(JSON_OUT, 'w') as f:
    f.write(data)
    
print("DONE")