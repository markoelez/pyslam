import numpy as np

def FtoRt(F, K):
  E = np.dot(np.dot(np.linalg.inv(K).transpose(), F), K) 

  diag_110 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
  W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

  U, S, V = np.linalg.svd(E)
  
  E = np.dot(np.dot(U, diag_110), V.transpose())

  U, S, V = np.linalg.svd(E)

  if np.linalg.det(U) < 0:
      U *= -1.0
  if np.linalg.det(V) < 0:
      V *= -1.0

  R = np.dot(np.dot(U, W), V.transpose())
  if np.sum(R.diagonal()) < 0:
      R = np.dot(np.dot(U, W.transpose()), V.transpose())

  t = U[:, 2]

  print(R)
  print(t)
  print('-----------------------------------------------')

  return R, t
    
