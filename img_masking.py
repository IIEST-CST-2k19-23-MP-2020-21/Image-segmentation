# functions to perform different operations on images


# libraries required
# Numpy

# function for run-length encoding

def runline_encoding(mask):
  # the pixels or mask should be passed as a numpy arrray
  # this function will return a run-line encoding for the mask
  # first step - flatten the numpy array
  flat_mask = mask.flatten()
  
  # padding the flattened mask before slicing
  padded_mask = np.concatenate([[0], flat_mask, [0]])
  
  run_arr = np.where(padded_mask[1:] != padded[:-1])[0]
  run_arr += 1
  run_arr[1::2] -= run_arr[0::2]
  encoding = ' '.join(str(run) for run in run_arr)
  return encoding
  
 # function for rebuilding mask
 
 def mask_rebuild(encoding, shape):
  # the encoded string should be passed along with the size of the mask i.e. width and height 
  # pass a tuple
  # the function returns the mask of the given size as a numpy array
  run_arr = np.asarray([int(run) for run in encoding.split(' ')])
  # perform the reverse operation
  run_arr[1::2] += run_arr[0::2]
  run_arr -= 1
  starting, ending = run_arr[0::2], run_arr[1::2]
  
  height, width = shape
  mask = np.zeros(height*width, dtype = uint8)
  for start, end in zip(starting, ending):
    mask[start:end] = 1
  return mask.reshape(shape)
  
