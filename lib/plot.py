from scipy.io import *
import numpy as np
import matplotlib.pyplot as plt
import os

def get_diff_x_rawdata(root_dir:str,exp_object:str,x_list:list):
  raw_datas = {}
  for x in x_list:
    dir_path = os.path.join(root_dir,exp_object,x)
    files = os.listdir(dir_path)
    files = sorted(files)
    
    file_path = os.path.join(dir_path,files[0])

    data = loadmat(file_path)['data']

    raw_datas[x] = data
  
  return raw_datas

def plot_radar_data(ax,raw_data:np.array,title:str):
  c = ax.imshow(np.abs(raw_data), cmap='viridis', aspect='auto', origin='lower')

  cbar = plt.colorbar(c)
  cbar.set_label('Value')

  ax.set_xlabel('Distance')
  ax.set_ylabel('Time')
  
  ax.set_title(title)

def plot_iq_data(ax,bin_data:np.array,color:str,text:str):
  real_part = np.real(bin_data)
  imag_part = np.imag(bin_data)

  ax.scatter(real_part,imag_part,color=color,marker = '.')


  ax.text(real_part[0],imag_part[0],text,fontsize=15)

