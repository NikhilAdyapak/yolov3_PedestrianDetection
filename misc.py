f = open('coco.names','r')

names = f.read().split()
print(len(names))
print(names[0],names[92])