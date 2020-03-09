import json
graph = json.load(open("ppi-G.json"))
out = open("ppi-edge-list.txt","w+")
x = graph['links']
for i in range(len(x)):
    print(x[i]['source'],"\t",x[i]['target'], file=out)
out.close()
