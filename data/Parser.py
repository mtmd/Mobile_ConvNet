import sys
from os import listdir
from os.path import isfile, join
    
path = 'Logs'
files = [f for f in listdir(path) if isfile(join(path, f))]


total_dic = {}
for file in files:
  if file == 'parallel.txt':
    continue
  dic = {}
  f = open(join(path, file), 'rU')
  for line in f:
    line = line[:-1]
    line = line.replace('\t', ' ')
    splited_str = line.split(' ')
    if splited_str[0] in dic:
      tmp = dic[splited_str[0]]
      tmp.append(int(splited_str[-1]))
      dic[splited_str[0]] = tmp
    else:
      tmp = [int(splited_str[-1])]
      dic[splited_str[0]] = tmp
  keys = dic.keys()
  for key in keys:
    tmp = dic[key]
    tmp = sorted(tmp)
    avg = sum(tmp[1:-1])
    avg = avg * 1.0 / (len(tmp) - 2)
    if key in total_dic:
      tmp = total_dic[key]
      tmp.append(avg)
      total_dic[key] = tmp
    else:
      tmp = [avg]
      total_dic[key] = tmp
  f.close()
keys = total_dic.keys()
keys = sorted(keys)
mn = []
for key in keys:
  tmp = total_dic[key]
  mn.append(min(tmp))
  print key, min(tmp)
print sum(mn)
