# encoding=UTF-8

"""
@author: pp on 7/20/16 4:25 PM
@email: ppsunrise99@gmail.com
@step:
@function:

"""



import numpy

import theano
from theano import config
from theano import tensor
from sklearn.decomposition import PCA
theano.config.floatX = 'float32'
config.floatX = 'float32'
# def ortho_weight(ndim):  # 随机生成方阵 奇异值分解
#     W = numpy.random.randn(ndim, ndim).astype(config.floatX)
#     u, s, v = numpy.linalg.svd(W)
#     return u.astype(config.floatX)
if __name__ == '__main__':
   def test(a):
       return a
   A = tensor.matrix('A')

   steps = A.shape[0]
   result,update = theano.scan(fn=test,sequences=[A],n_steps=steps)
   add = theano.function([A],result)
   aa = numpy.array([[1,2],[3,4],[5,6]]).astype(config.floatX)

   rr = add(aa)[0]
   print rr

