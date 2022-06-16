import json
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

x,y,w,h = 1,2,3,4
d1 = dict()
d2 = dict()
d1 = {'x':x,'y':y,'w':w,'h':h}
x,y,w,h = 11,12,13,14
d2 = {'x':x,'y':y,'w':w,'h':h}
x,y,w,h = 11,12,13,14
'''with open('data.json','w') as f:
    json.dump({'w':w},f,ensure_ascii=False)
    json.dump({'y':y},f,ensure_ascii=False)
'''
data = (d1,d2)

with io.open('data.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(to_unicode(str_))

data = [d1,d2,d1,d2]
data = tuple(data)
print(data)