import os.path

from source import *
cwd = os.getcwd()
newpath = os.path.join(cwd,'data','07-20-small-dt','IMMEC_history_45V_1.0sec.pkl')
#newpath = os.path.join(os.path.dirname(cwd),'Cantoni.pkl')
# motordict = read_motordict(newpath)
with open(newpath, 'rb') as f:
    data = pkl.load(f)

#print(data["d_air"])
print(data.keys()) # the UMP is F_em
