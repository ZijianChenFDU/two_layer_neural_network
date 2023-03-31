import pickle
import os

def save_network(network,filename):
  '''保存训练好的网路'''
  if os.path.exists(filename):
    (name, suffix) = os.path.splitext(filename)
    filename = name + "(1)" + suffix
  f=open(filename,'wb')          
  pickle.dump(network,f)          
  f.close()                  
  print("\n The netowrk has been saved in "+filename+"!")

def load_network(filename):
  '''加载训练好的网络'''
  try:
    f=open(filename,'rb')
    network=pickle.load(f)
    f.close()
    return network
  
  except EOFError:
    return ""