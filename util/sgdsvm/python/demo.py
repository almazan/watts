import numpy as np
from  sgd_wrap import sgd_cv_params_t
from sgd_wrap import sgdalbert_train_cv_oneclass
from ctypes import POINTER, c_int32, c_float
# Load data
Xtrain = np.loadtxt("../datasets/ijcnn1/ijcnn1.tr.data", dtype=np.float32)
Ltrain = np.loadtxt("../datasets/ijcnn1/ijcnn1.tr.labels", dtype = np.int32)
Xvalid = np.loadtxt("../datasets/ijcnn1/ijcnn1.val.data", dtype=np.float32)
Lvalid = np.loadtxt("../datasets/ijcnn1/ijcnn1.val.labels", dtype = np.int32)
Xtest = np.loadtxt("../datasets/ijcnn1/ijcnn1.t.data", dtype=np.float32)
Ltest = np.loadtxt("../datasets/ijcnn1/ijcnn1.t.labels", dtype = np.int32)

# Prepare parameters
etas = np.array([2,1,1e-1,1e-2], dtype=np.float32)
lbds = np.array([ 1e-2,1e-3, 1e-4,1e-5,1e-6,1e-7,1e-8], dtype=np.float32)
betas = np.array( [32,64,128,256,512], dtype = np.int32)
biases = np.array([1], dtype = np.float32)

params = sgd_cv_params_t()
params.eta0s = etas.ctypes.data_as(POINTER(c_float))
params.netas = len(etas)
params.lbds = lbds.ctypes.data_as(POINTER(c_float))
params.nlambdas = len(lbds)
params.betas = betas.ctypes.data_as(POINTER(c_int32))
params.nbetas = len(betas)
params.bias_multipliers = biases.ctypes.data_as(POINTER(c_float))
params.nbias_multipliers = len(biases)
params.epochs = 100
params.eval_freq = 2
params.t = 0
params.weightPos = 1
params.weightNeg = 1
[W,Bias,PlattsA,PlatsB, info] = sgdalbert_train_cv_oneclass(1,  Xtrain,Ltrain,Xvalid,Lvalid,  params)
print "map on validation: %.2f"%(info.acc)
print "eta0: %.6f lbd: %.8f beta: %d epoch: %d"%(info.eta0, info.lbd, info.beta, info.epoch)

# Val:

print "Results on validation:"
s = np.dot(Xvalid,W) + Bias
r1 = (np.where(Lvalid[s>=0]==1)[0]).shape[0]/float(np.where(Lvalid==1)[0].shape[0])
p1 = (np.where(Lvalid[s>=0]==1)[0]).shape[0]/float(np.where(s>=0)[0].shape[0])
r0 = (np.where(Lvalid[s<0]==-1)[0]).shape[0]/float(np.where(Lvalid==-1)[0].shape[0])
p0 = (np.where(Lvalid[s<0]==-1)[0]).shape[0]/float(np.where(s<0)[0].shape[0])
print "p1: %.2f r1: %.2f p-1: %.2f r-1: %.2f"%(100*p1,100*r1,100*p0,100*r0)


# Test:
print "Results on test:"
s = np.dot(Xtest,W) + Bias
r1 = (np.where(Ltest[s>=0]==1)[0]).shape[0]/float(np.where(Ltest==1)[0].shape[0])
p1 = (np.where(Ltest[s>=0]==1)[0]).shape[0]/float(np.where(s>=0)[0].shape[0])
r0 = (np.where(Ltest[s<0]==-1)[0]).shape[0]/float(np.where(Ltest==-1)[0].shape[0])
p0 = (np.where(Ltest[s<0]==-1)[0]).shape[0]/float(np.where(s<0)[0].shape[0])
print "p1: %.2f r1: %.2f p-1: %.2f r-1: %.2f"%(100*p1,100*r1,100*p0,100*r0)

