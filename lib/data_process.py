import numpy as np
from .plot import *
def remove_other_object(root_dir:str,raw_datas:dict,x_list:list):
  no_object_datas = get_diff_x_rawdata(root_dir,'no_object',x_list)
  # no_object_datas = phase_noise_correct(no_object_datas,x_list)
  processed_datas = {}
  for x in x_list:
    raw_data = raw_datas[x]
    no_object_data = no_object_datas[x]
    ref_value = np.mean(no_object_data,axis=0)
    processed_data = raw_data - ref_value
    processed_datas[x] = processed_data

  return processed_datas

def concat_radar_data(datas:np.array,x_list:list):
  concat_data = np.empty((0,96))
  for x in x_list:
    data = datas[x]
    concat_data = np.vstack((concat_data,data))
  return concat_data

def phase_noise_correct(raw_datas:dict,x_list:list):

  correct_datas = {}

  for x in x_list:
    correct_data = raw_datas[x]
    bin0_data = correct_data[:,0]
    bin0_phase = np.angle(bin0_data)
    mean_phase = np.mean(bin0_phase)

    for frame in range(correct_data.shape[0]):
      diff_phase = np.angle(correct_data[frame,0]) - mean_phase
      frame_data = correct_data[frame,:]
      frame_data = frame_data * np.exp(1j * diff_phase)
      correct_data[frame,:] = frame_data
    correct_datas[x] = correct_data
  return correct_datas

    