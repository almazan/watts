import numpy as np
from numpy import ctypeslib
import ctypes
from ctypes import *
lsgd = ctypes.cdll.LoadLibrary("../libsgdsvm.so")

lsgd.Platts.argtypes = [np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*scores
                        ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                ,c_int # l    
                , POINTER(c_float) #PlattsA
                , POINTER(c_float)] #PlattsB

def LearnPlatts(scores, Lval):
    PlattsA= c_float()
    PlattsB= c_float()
    lsgd.Platts(scores, Lval.T[0], len(Lval), byref(PlattsA), byref(PlattsB))
    return PlattsA.value, PlattsB.value


class sgd_params_t(Structure):
    _fields_ = [
            ("eta0", c_float),
            ("lbd", c_float),
            ("beta", c_int32),
            ("bias_multiplier", c_float),
            ("epochs", c_int32),
            ("eval_freq", c_int32),
            ("t", c_int32),
            ("weightPos", c_float),
            ("weightNeg", c_float)]


class sgd_cv_params_t(Structure):
    _fields_ = [
            ("eta0s", POINTER(c_float)),
            ("netas", c_int32),
            ("lbds", POINTER(c_float)),
            ("nlambdas", c_int32),
            ("betas", POINTER(c_int32)),
            ("nbetas", c_int32),
            ("bias_multipliers", POINTER(c_float)),
            ("nbias_multipliers", c_int32),
            ("epochs", c_int32),
            ("eval_freq", c_int32),
            ("t", c_int32),
            ("weightPos", c_float),
            ("weightNeg", c_float)]




class sgd_output_info_t(Structure):
    _fields_ = [
            ("eta0", c_float),
            ("lbd", c_float),
            ("beta", c_int32),
            ("bias_multiplier", c_float),
            ("t", c_int32),
            ("updates", c_int32),
            ("epoch", c_int32),
            ("acc", c_float),
            ("weightPos", c_float),
            ("weightNeg", c_float)]







class pq_info_t(Structure):
    _fields_ = [
            ("nsq", c_int32),
            ("ksq", c_int32),
            ("dsq", c_int32),
            ("centroids", POINTER(c_float))]



lsgd.sgd_train_class_cv.argtypes = [c_int # c
                           ,c_int #Ntrain
                           ,c_int #d
				, POINTER(c_float) #Xtrain
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                           ,c_int #Nval
			   , POINTER(c_float) #Xval
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_cv_params_t)
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,POINTER(c_float) #*B
                           ,POINTER(c_float) #*PlattsA
                           ,POINTER(c_float) #*PlattsB
                           ,POINTER(sgd_output_info_t)]


lsgd.sgd_train_class_cv_pq.argtypes = [c_int # c
                , POINTER(pq_info_t)
                           ,c_int #Ntrain
                           ,c_int #d
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xtrain_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                           ,c_int #Nval
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xval_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_cv_params_t)
                           ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,POINTER(c_float) #*B
                           ,POINTER(c_float) #*PlattsA
                           ,POINTER(c_float) #*PlattsB
                           ,POINTER(sgd_output_info_t)]


lsgd.sgd_train_class_pq.argtypes = [c_int # c
                , POINTER(pq_info_t)
                           ,c_int #Ntrain
                           ,c_int #d                                                      
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xtrain_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Ltrain
                ,c_int #Nval
                           ,np.ctypeslib.ndpointer(dtype=c_uint8, ndim=1) #Xval_pq
                           ,np.ctypeslib.ndpointer(dtype= c_int, ndim=1) #Lval  
                           ,POINTER(sgd_params_t)
               ,np.ctypeslib.ndpointer(dtype=c_float, ndim=1) #*W
                           ,POINTER(c_float) #*B
                           ,POINTER(sgd_output_info_t)]



def sgdalbert_train_cv_oneclass(cls,  Xtrain,Ltrain,Xval,Lval,  params):
    d = Xtrain.shape[1]
    ntrain = len(Ltrain)
    nval = len(Lval)
    W = np.zeros(d,dtype = np.float32)
    bias_tmp=c_float()
    plattsA_tmp=c_float()
    plattsB_tmp=c_float()    
    info = sgd_output_info_t()
    lsgd.sgd_train_class_cv(cls,ntrain,d,  Xtrain.ravel().ctypes.data_as(POINTER(c_float)),Ltrain,nval, Xval.ravel().ctypes.data_as(POINTER(c_float)),Lval,byref(params),W, byref(bias_tmp), byref(plattsA_tmp), byref(plattsB_tmp), byref(info))
    return W,bias_tmp.value,plattsA_tmp.value,plattsB_tmp.value,info


def sgdalbert_train_cv_pq_oneclass(cls,pq,  Xtrain_pqcodes,Ltrain,Xval_pqcodes,Lval,  params):
    d = pq.nsq*pq.dsq
    ntrain = len(Ltrain)
    nval = len(Lval)
    W = np.zeros(d,dtype = np.float32)
    bias_tmp=c_float()
    plattsA_tmp=c_float()
    plattsB_tmp=c_float()    
    info = sgd_output_info_t()
    lsgd.sgd_train_class_cv_pq(cls,byref(pq),  ntrain,d,  Xtrain_pqcodes,Ltrain.T[0],nval, Xval_pqcodes,Lval.T[0],byref(params),W, byref(bias_tmp), byref(plattsA_tmp), byref(plattsB_tmp), byref(info))
    return W,bias_tmp.value,plattsA_tmp.value,plattsB_tmp.value,info

