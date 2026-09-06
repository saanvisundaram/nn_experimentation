# people_politics.py
# neural network, scratch Python

import numpy as np

class NeuralNetwork:

  def __init__(self, num_in, num_hid, num_out, seed):
    self.ni = num_in
    self.nh = num_hid
    self.no = num_out
	
    self.i_nodes = np.zeros(shape=self.ni, dtype=np.float32)
    self.h_nodes = np.zeros(shape=self.nh, dtype=np.float32)
    self.o_nodes = np.zeros(shape=self.no, dtype=np.float32)
	
    self.ih_weights = np.zeros(shape=(self.ni,self.nh),
      dtype=np.float32)
    self.ho_weights = np.zeros(shape=(self.nh,self.no),
      dtype=np.float32)
	
    self.h_biases = np.zeros(shape=self.nh, dtype=np.float32)
    self.o_biases = np.zeros(shape=self.no, dtype=np.float32)

    self.ih_grads = np.zeros((self.ni, self.nh),
      dtype=np.float32)
    self.hb_grads = np.zeros(self.nh, dtype=np.float32)
    self.ho_grads = np.zeros((self.nh, self.no),
      dtype=np.float32)
    self.ob_grads = np.zeros(self.no, dtype=np.float32)
	
    self.rnd = np.random.RandomState(seed)
    self.init_weights()

# -----------------------------------------------------------

  def init_weights(self):
    num_wts = (self.ni * self.nh) + self.nh + \
      (self.nh * self.no) + self.no
    wts = np.zeros(shape=num_wts, dtype=np.float32)
    lo = -0.01; hi = 0.01
    for i in range(len(wts)):
      wts[i] = (hi - lo) * self.rnd.random() + lo
    self.set_weights(wts)

# -----------------------------------------------------------

  def set_weights(self, weights):
    idx = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.ih_weights[i][j] = weights[idx]
        idx += 1
    for j in range(self.nh):
      self.h_biases[j] = weights[idx]
      idx += 1
    for j in range(self.nh):
      for k in range(self.no):
        self.ho_weights[j][k] = weights[idx]
        idx += 1
    for k in range(self.no):
      self.o_biases[k] = weights[idx]
      idx += 1

# -----------------------------------------------------------

  def get_weights(self):
    # order: ih_wts, h_biases, ho_wts, o_biases
    num_wts = (self.ni * self.nh) + self.nh + \
      (self.nh * self.no) + self.no
    result = np.zeros(num_wts, dtype=np.float32)
    p = 0
    for i in range(self.ni):
      for j in range(self.nh):
        result[p] = self.ih_weights[i][j]
        p += 1
    for j in range(self.nh):
      result[p] = self.h_biases[j]
      p += 1
    for j in range(self.nh):
      for k in range(self.no):
        result[p] = self.ho_weights[j][k]
        p += 1
    for k in range(self.no):
      result[p] = self.o_biases[k]
      p += 1
    return result

# -----------------------------------------------------------

  def compute_outputs(self, x):
    h_sums = np.zeros(self.nh, dtype=np.float32)
    o_sums = np.zeros(self.no, dtype=np.float32)
    # copy x into i_nodes to avoid by-ref errors
    for i in range(len(x)):
      self.i_nodes[i] = x[i]

    for j in range(self.nh):
      for i in range(self.ni):
        h_sums[j] += self.i_nodes[i] * self.ih_weights[i][j]
      h_sums[j] += self.h_biases[j]
      self.h_nodes[j] = np.tanh(h_sums[j])

    for k in range(self.no):
      for j in range(self.nh):
        o_sums[k] += self.h_nodes[j] * self.ho_weights[j][k]
      o_sums[k] += self.o_biases[k]

    # apply softmax
    soft_out = self.softmax(o_sums)
    for k in range(self.no):
      self.o_nodes[k] = soft_out[k]

    # create copy of o_nodes for explicit return
    result = np.zeros(shape=self.no, dtype=np.float32)
    for k in range(self.no):
      result[k] = self.o_nodes[k]
	  
    return result

# -----------------------------------------------------------
    
  @staticmethod	  
  def softmax(o_sums):
    result = np.zeros(shape=len(o_sums), dtype=np.float32)
    m = np.max(o_sums)
    divisor = 0.0
    for k in range(len(o_sums)):
       divisor += np.exp(o_sums[k] - m)
    for k in range(len(result)):
      result[k] =  np.exp(o_sums[k] - m) / divisor
    return result

# -----------------------------------------------------------

  def zero_out_grads(self):
    for i in range(self.ni):
      for j in range(self.nh):
        self.ih_grads[i][j] = 0.0
    for j in range(self.nh):  
      self.hb_grads[j] = 0.0
    for j in range(self.nh):
      for k in range(self.no):
        self.ho_grads[j][k] = 0.0
    for k in range(self.no):
      self.ob_grads[k] = 0.0

# -----------------------------------------------------------

  def accum_grads(self, y):
    # y is target vector
    o_signals = np.zeros(self.no, dtype=np.float32)
    h_signals = np.zeros(self.nh, dtype=np.float32)

    # 1. compute output node scratch signals 
    for k in range(self.no):
      derivative = 1.0  # CEE
      # derivative =
      #  self.oNodes[k] * (1 - self.o_nodes[k]) # MSE
      o_signals[k] = derivative * \
        (self.o_nodes[k] - y[k])  # CEE

    # 2. accum hidden-to-output gradients 
    for j in range(self.nh):
      for k in range(self.no):
        self.ho_grads[j][k] += o_signals[k] * \
          self.h_nodes[j]

    # 3. accum output node bias gradients
    for k in range(self.no):
      self.ob_grads[k] += o_signals[k] * 1.0 

    # 4. compute hidden node signals
    for j in range(self.nh):
      sum = 0.0
      for k in range(self.no):
        sum += o_signals[k] * self.ho_weights[j][k]

      derivative = \
        (1 - self.h_nodes[j]) * \
        (1 + self.h_nodes[j])  # assumes tanh
      h_signals[j] = derivative * sum

    # 5. accum input-to-hidden gradients
    for i in range(self.ni):
      for j in range(self.nh):
        self.ih_grads[i][j] += \
          h_signals[j] * self.i_nodes[i]

    # 6. accum hidden node bias gradients
    for j in range(self.nh):
      self.hb_grads[j] += h_signals[j] * 1.0

# -----------------------------------------------------------

  def update_weights(self, lrn_rate):
    # assumes all gradients computed
    # 1. update input-to-hidden weights
    for i in range(self.ni):
      for j in range(self.nh):
        delta = -1.0 * lrn_rate * self.ih_grads[i][j]
        self.ih_weights[i][j] += delta

    # 2. update hidden node biases
    for j in range(self.nh):
      delta = -1.0 * lrn_rate * self.hb_grads[j]
      self.h_biases[j] += delta

    # 3. update hidden-to-output weights
    for j in range(self.nh):
      for k in range(self.no):
        delta = -1.0 * lrn_rate * self.ho_grads[j][k]
        self.ho_weights[j][k] += delta

    # 4. update output node biases
    for k in range(self.no):
      delta = -1.0 * lrn_rate * self.ob_grads[k]
      self.o_biases[k] += delta

# -----------------------------------------------------------

  def train(self, train_x, train_y, lrn_rate, bat_size,
    max_epochs):
    n = len(train_x)                  # like 200
    batches_per_epoch = n // bat_size # like 20
    freq = max_epochs / 10            # progress
    indices = np.arange(n)

    for epoch in range(max_epochs): 
      self.rnd.shuffle(indices)
      ptr = 0   # points into indices
      for bat_idx in range(batches_per_epoch): # 0, 1, .. 19
        for i in range(bat_size):  # 0 . . 9
          ii = indices[ptr]; ptr += 1
          x = train_x[ii]
          y = train_y[ii]
          self.compute_outputs(x)  # into self.o_nodes
          self.accum_grads(y)

        self.update_weights(lrn_rate)
        self.zero_out_grads()  # prep for next batch
 
      if epoch % freq == 0:
        # mse = 
        # self.mean_sq_err(train_x, train_y)
        mcee = self.mean_cross_ent_err(train_x, train_y)
        acc = self.accuracy(train_x, train_y)
        s1 = "epoch: %5d" % epoch
        s2 = "   MCEE = %8.4f" % mcee
        s3 = "   acc = %8.4f" % acc
        print(s1 + s2 + s3)

# -----------------------------------------------------------

  def mean_cross_ent_err(self, data_x, data_y):
    sum_cee = 0.0  # cross entropy errors
    for i in range(len(data_x)):
      x = data_x[i]
      y = data_y[i]  # target like (0, 1, 0)
      oupt = self.compute_outputs(x)
      idx = np.argmax(y)   # find loc of 1 in target
      sum_cee += np.log(oupt[idx])
    sum_cee *= -1
    return sum_cee / len(data_x)


# -----------------------------------------------------------

  def mean_sq_err(self, data_x, data_y):
    sum_se = 0.0
    for i in range(len(data_x)):
      x = data_x[i]
      y = data_y[i]   # target output like (0, 1, 0)
      oupt = self.compute_outputs(x)  # (0.23, 0.66, 0.11)
      for k in range(self.no):
        err = y[k] - oupt[k]  # target - computed

    return sum_se / len(data_x)   # consider Root MSE

# -----------------------------------------------------------

  def accuracy(self, data_x, data_y):
    nc = 0; nw = 0;
    for i in range(len(data_x)):
      x = data_x[i]
      y = data_y[i] ;  # target like (0, 1, 0)
      oupt = self.compute_outputs(x)
      computed_idx = np.argmax(oupt)
      target_idx = np.argmax(y)
      if computed_idx == target_idx:
        nc += 1
      else:
        nw += 1
    return nc / (nc + nw)

# -----------------------------------------------------------

  def confusion_matrix(self, data_x, data_y):
    result = np.zeros((self.no,self.no), dtype=np.float32)
    for i in range(len(data_x)):
      x = data_x[i]
      y = data_y[i]  # target like (0, 1, 0)
      oupt = self.compute_outputs(x)  # probs form
      target_k = np.argmax(y)
      pred_k = np.argmax(oupt)
      result[target_k][pred_k] += 1
    return result;

# -----------------------------------------------------------

  def show_confusion(self, cm):
    n = len(cm)
    for i in range(n):
      print("actual %2d: " % i, end="")
      for j in range(n):
        print("%8d" % cm[i][j], end="")
      print("")

# -----------------------------------------------------------

  def save_weights(self, fn):
    # write weights as single comma-delimied line
    wts = self.get_weights()
    n = len(wts)
    ofs = open(fn, "w")
    for i in range(n):
      w = wts[i]
      ofs.write("%0.4f" % w)
      if i != n-1:
        ofs.write(",")
    ofs.write("\n")
    ofs.close()

# -----------------------------------------------------------

  def load_weights(self, fn):
    ifs = open(fn, "r")
    s = ifs.readline()
    tokens = s.split(",")
    wts = np.zeros(len(tokens), dtype=np.float32)
    for i in range(len(wts)):
      wts[i] = float(tokens[i])
    ifs.close()
    self.set_weights(wts)

# -----------------------------------------------------------

  @staticmethod	
  def vec_to_onehot(data_y, n):
    # convert ordinal (0,1,2 . .) to one-hot
    rows = len(data_y)
    cols = n
    result = np.zeros((rows,cols), dtype=np.float32)
    for i in range(rows):
      k = data_y[i]   # 0,1,2 . .
      result[i][k] = 1.0;  # [ 0.0  1.0  0.0]
    return result;

# -----------------------------------------------------------
# -----------------------------------------------------------

def main():
  print("\nBegin NN using raw Python demo ")
  # 1. load data
  #  1, 0.24, 1, 0, 0, 0.2950, 2
  # -1, 0.39, 0, 0, 1, 0.5120, 1

  print("\nLoading data into memory ")
  train_file = "people_train.txt"
  test_file = "people_test.txt"

  train_x = np.loadtxt(train_file, usecols=[0,1,2,3,4,5],
    delimiter=",", comments="#", dtype=np.float32)
  train_y = np.loadtxt(train_file, usecols=6,
    delimiter=",", comments="#", dtype=np.int64)
  train_y = NeuralNetwork.vec_to_onehot(train_y, 3)

  test_x = np.loadtxt(test_file, usecols=[0,1,2,3,4,5],
    delimiter=",", comments="#", dtype=np.float32)
  test_y = np.loadtxt(test_file, usecols=6,
    delimiter=",", comments="#", dtype=np.int64)
  test_y = NeuralNetwork.vec_to_onehot(test_y, 3)

  # 2. create network
  print("\nCreating 6-10-3 tanh, softmax CEE NN ")
  nn = NeuralNetwork(6, 10, 3, seed=0)

  # 3. train network
  lrn_rate = 0.01
  max_epochs = 1000
  print("\nSetting learn rate = 0.01 ")
  print("Setting batch size = 10 ")
  print("Setting max epochs = 1000 ")
  print("\nStarting training ")
  nn.train(train_x, train_y, lrn_rate, 10, max_epochs)
  print("Training complete ")

  # 4. evaluate model
  train_acc = nn.accuracy(train_x, train_y)
  test_acc = nn.accuracy(test_x, test_y)
  print("\nAccuracy on training data = %0.4f" % train_acc)
  print("Accuracy on test data = %0.4f" % test_acc)

  print("\nConfusion matrix training data: ")
  cm = nn.confusion_matrix(train_x, train_y)
  nn.show_confusion(cm)

  # 5. save trained model
  print("\nSaving trained weights to file ")
  nn.save_weights("politics_weights.txt")

  # nn = NeuralNetwork(6, 25, 3, seed=0)
  # nn.load_weights(".\\Models\\politics_weights.txt")

  # 6. use trained model
  print("\nPredict for M 46 Oklahoma $66,400 ")
  x = np.array([-1, 0.46, 0, 0, 1, 0.6640], dtype=np.float32)
  pred_probs = nn.compute_outputs(x)
  print("\nPredicted pseudo-probabilities: ")
  np.set_printoptions(precision=4, floatmode='fixed',
    suppress=True)
  print(pred_probs)

  print("\nEnd demo ")

if __name__ == "__main__":
  main()