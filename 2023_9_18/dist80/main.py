from scipy.io import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/muxin/hdd/datas/exp/')
from lib import *

exp_object_list = ['no_object','paper_box']

color_list = ['b','g','r','c','m']

x_list = ['-20','-10','0','10','20']

root_dir = "/home/muxin/hdd/datas/exp/2023_9_18/dist80"

sub_plot_row = 2
sub_plot_column = 3 

def dif_x(exp_object:str):
  raw_datas = get_diff_x_rawdata(root_dir,exp_object,x_list)

  test_fig,test_ax = plt.subplots(figsize = (18,10))

  raw_data_x0 = raw_datas['0']

  frame0_data = raw_data_x0[0,:]
  frame1_data = raw_data_x0[100,:]

  test_ax.plot(np.angle(frame0_data),color='r')
  test_ax.plot(np.angle(frame1_data),color='b')
  test_ax.set_title('adjust frame')

  # raw_datas =  phase_noise_correct(raw_datas,x_list)
  
  raw_datas_fig,raw_datas_axes = plt.subplots(sub_plot_row,sub_plot_column, figsize = (18,10))

  plt.suptitle("raw data for {} in different x".format(exp_object))

  for i,x in enumerate(x_list):
    title = "raw data for {} in x={}".format(exp_object,x)

    plot_radar_data(raw_datas_axes[i//sub_plot_column][i%sub_plot_column],raw_datas[x],title)

  concat_raw_fig,concat_raw_ax = plt.subplots(figsize = (10,18))

  title = "concat raw data for {} in different x".format(exp_object)

  concat_raw_data = concat_radar_data(raw_datas,x_list)

  plot_radar_data(concat_raw_ax,concat_raw_data,title)

  if exp_object == "no_object":
    return

  processed_datas = remove_other_object(root_dir,raw_datas,x_list)

  processed_datas_fig,processed_datas_axes = plt.subplots(sub_plot_row,sub_plot_column, figsize = (18,10))

  plt.suptitle("processed data for {} in different x".format(exp_object))

  for i,x in enumerate(x_list):
    title = "processed data for {} in x={}".format(exp_object,x)

    plot_radar_data(processed_datas_axes[i//sub_plot_column][i%sub_plot_column],processed_datas[x],title)

  concat_pro_fig,concat_pro_ax = plt.subplots(figsize = (10,14))

  title = "concat processed data for {} in different x".format(exp_object)

  concat_pro_data = concat_radar_data(processed_datas,x_list)

  plot_radar_data(concat_pro_ax,concat_pro_data,title)

  iq_fig,iq_ax = plt.subplots(figsize = (12,12))

  iq_ax.set_xlabel('In-phase (I)')
  iq_ax.set_ylabel('Quadrature (Q)')
  iq_ax.set_title('Complex Signal in IQ Domain')

  max_bins = {}

  for i,x in enumerate(x_list):
    processed_data = processed_datas[x]

    mean_val = np.mean(processed_data,axis=0)

    abs_val = np.abs(mean_val)

    max_bin = np.argmax(abs_val)

    max_bins[x] = max_bin

    text = "x = {},bin = {}".format(x,max_bin)

    plot_iq_data(iq_ax,processed_data[:,max_bin],color_list[i],text)

  iq1_fig,iq1_ax = plt.subplots(figsize = (12,12))

  iq1_ax.set_xlabel('In-phase (I)')
  iq1_ax.set_ylabel('Quadrature (Q)')
  iq1_ax.set_title('Complex Signal in IQ Domain')

  for i,x in enumerate(x_list):
    raw_data = raw_datas[x]

    text = "x = {},bin = {}".format(x,max_bins[x])

    plot_iq_data(iq1_ax,raw_data[:,max_bins[x]],color_list[i],text)

  datas = phase_noise_correct(raw_datas,x_list)

  iq2_fig,iq2_ax = plt.subplots(figsize = (12,12))

  iq2_ax.set_xlabel('In-phase (I)')
  iq2_ax.set_ylabel('Quadrature (Q)')
  iq2_ax.set_title('Complex Signal in IQ Domain')

  for i,x in enumerate(x_list):
    data = datas[x]

    text = "x = {},bin = {}".format(x,max_bins[x])

    plot_iq_data(iq2_ax,data[:,max_bins[x]],color_list[i],text)






if __name__ == "__main__":
  for exp_object in exp_object_list:
    dif_x(exp_object)
  plt.show()