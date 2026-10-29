import sys; sys.path.insert(0,'.')
import numpy as np
from src.label_coverage import *
rng=np.random.default_rng(0)
names=[f"L{i}" for i in range(14)]
p=np.array([.3,.2,.1,.01,.05,.04,.03,.08,.02,.15,.005,.002,.2,.4])
pool=rng.random((20000,14))<p
pool[:,11]=False  # nhãn 11 không có ở đâu
train=rng.random((6400,14))<p; train[:,10]=False   # nhãn 10: ngoại vi train
test=rng.random((3000,14))<p; test[:,10]=False; test[:,3]=False
t=coverage_table(train,test,names,30,pool); print(t[['label','train_pos','test_pos','status']].to_string())
print('fails',fails(t))
idx,info=select_expert_test(pool,3000,30,train,42)
t2=coverage_table(train,pool[idx],names,30,pool); print(info); print(t2[['label','train_pos','test_pos','status']].to_string()); print('fails2',fails(t2))
