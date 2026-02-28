import os,sys
d=os.open(os.devnull,os.O_WRONLY);os.dup2(d,2);os.close(d)
import spliceai,tensorflow as tf
from keras.saving import load_model
K=[load_model(spliceai.__path__[0]+f"/models/spliceai{i}.h5",compile=False)for i in range(1,6)]
def f(s):
 x=tf.one_hot([4]*5000+[{65:0,67:1,71:2,84:3,85:3}.get(c,4)for c in s.encode()]+[4]*5000,5)[:,:4][None]
 y=0
 for k in K:
  o=k(x,training=False);o=o[0]if isinstance(o,(list,tuple))else o
  y+=o[0]
 y/=5
 return tf.math.bincount(tf.searchsorted((.001,.05,.2,.35,.5,.8),tf.concat((y[:,1],y[:,2]),0),side='right'),None,7).numpy()
s=''.join(sys.stdin.read().split()).upper()
if s:print(*f(s));print(*f(s[::-1]))
