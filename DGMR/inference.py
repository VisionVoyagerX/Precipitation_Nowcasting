#import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_pipeline import Dataset
from pathlib import Path
from dgmr import DGMR
from losses import Loss_hing_disc, Loss_hing_gen
import os
import matplotlib.pyplot as plt
from utils import *
import argparse
from tensorflow import keras
from matplotlib import animation
import matplotlib
import numpy as np
import cv2
from utils import *


def plot_animation_on_map_6(field, name=None, figsize=None,
                            vmin=0, vmax=10, cmap="jet", **imshow_args):
    background = cv2.imread("NLD_divs.png")

    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.figsize"] = [16., 7.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['image.cmap'] = 'viridis'
    
    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(5, 5))
    fig.set_figheight(4)
    fig.set_figwidth(12)
    fig.text(0., 0.5, 'DGMR', va='center', rotation='vertical')
    #ax = plt.axes()
    # ax.set_title(str(name))
    plt.close()  # Prevents extra axes being plotted below animation

    ax[0].set_title('Ensemble 1', fontsize=12)
    ax[1].set_title('Ensemble 2', fontsize=12)
    ax[2].set_title('Ensemble 3', fontsize=12)
    ax[3].set_title('Ensemble 4', fontsize=12)
    ax[4].set_title('Ensemble 5', fontsize=12)
    ax[5].set_title('Ensemble 6', fontsize=12)

    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()
    ax[4].set_axis_off()
    ax[5].set_axis_off()
    
    # ax.set_title(str(name))
    plt.close()  # Prevents extra axes being plotted below animation
    #ax[0].imshow(background)
    #ax[1].imshow(background)
    #ax[2].imshow(background)
    #ax[3].imshow(background)
    #ax[4].imshow(background)
    #ax[5].imshow(background)
    img0 = ax[0].imshow(field[0][0, ...], vmin=vmin,
                        vmax=vmax, cmap=cmap, **imshow_args)
    img1 = ax[1].imshow(field[1][0, ...], vmin=vmin,
                        vmax=vmax, cmap=cmap, **imshow_args)
    img2 = ax[2].imshow(field[2][0, ...], vmin=vmin,
                        vmax=vmax, cmap=cmap, **imshow_args)
    img3 = ax[3].imshow(field[3][0, ...], vmin=vmin,
                        vmax=vmax, cmap=cmap, **imshow_args)
    img4 = ax[4].imshow(field[4][0, ...], vmin=vmin,
                        vmax=vmax, cmap=cmap, **imshow_args)
    img5 = ax[5].imshow(field[5][0, ...], vmin=vmin,
                        vmax=vmax, cmap=cmap, **imshow_args)

    def animate(i):
        img0.set_data(field[0][i, ...])
        img1.set_data(field[1][i, ...])
        img2.set_data(field[2][i, ...])
        img3.set_data(field[3][i, ...])
        img4.set_data(field[4][i, ...])
        img5.set_data(field[5][i, ...])
        return (img0, img1, img2, img3, img4, img5)
    anim = animation.FuncAnimation(
        fig, animate, frames=field[0].shape[0], blit=False)

    # saving
    #anim.save('./ens_vid/' + name + '.gif', writer='imagemagick', fps=60.0)
    FFwriter = animation.FFMpegWriter(fps=20)
    anim.save('./ens_vid/' + name + '.mp4', writer=FFwriter)

    return anim


def plot_animation(field, name=None, figsize=None,
                   vmin=0, vmax=10, cmap="jet", **imshow_args):
                   
    plt.rcParams['image.cmap'] = 'viridis'
    background = cv2.imread("NLD_divs.png")
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_axis_off()
    # ax.set_title(str(name))
    plt.close()  # Prevents extra axes being plotted below animation
    #ax.imshow(background)
    img = ax.imshow(field[0, ...], vmin=vmin,
                    vmax=vmax, cmap=cmap, **imshow_args)

    def animate(i):
        img.set_data(field[i, ...])
        return (img,)

    anim = animation.FuncAnimation(
        fig, animate, frames=field.shape[0], blit=False)

    # saving
    #anim.save('./deter_vid/' + name + '.gif', writer='imagemagick', fps=60.0)
    FFwriter = animation.FFMpegWriter(fps=20)
    anim.save('./deter_vid/' + name + '.mp4', writer=FFwriter)

    return anim


def video_predict(results_list, name):
    plot_animation_on_map_6(results_list, name)
    plot_animation(results_list[0], name)


def deter_img(results, name):

    plt.rcParams['image.cmap'] = 'viridis'
    background = cv2.imread("NLD_divs.png")
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(3)
    fig.set_figwidth(6)

    fig.text(0.04, 0.5, 'DGMR', va='center', rotation='vertical')

    #ax[0].imshow(background)
    ax[0].imshow(results[5])
    ax[0].set_axis_off()
    ax[0].set_title('T+30')

    #ax[1].imshow(background)
    ax[1].imshow(results[11])
    ax[1].set_axis_off()
    ax[1].set_title('T+60')

    #ax[2].imshow(background)
    ax[2].imshow(results[17])
    ax[2].set_axis_off()
    ax[2].set_title('T+90')
    fig.savefig('./img_deter/' + name + '.png')


def ens_img(results, name):

    plt.rcParams['image.cmap'] = 'viridis'
    background = cv2.imread("NLD_divs.png")
    fig, ax = plt.subplots(
        nrows=6, ncols=3, figsize=(10, 10))  # , figsize=(5, 5)
    matplotlib.rcParams.update({'font.size': 32})

    fig.set_figheight(30)
    fig.set_figwidth(15)
    fig.tight_layout()
    fig.text(0., 0.5, 'Ensembles', va='center', rotation='vertical')
    #fig.text(0.5, 0.0, 'Ensembles', ha='center', va='center')
    fig.suptitle('DGMR')
    fig.subplots_adjust(top=0.9)

    #ax[0, 0].imshow(background)
    ax[0, 0].imshow(results[0][5])
    ax[0, 0].set_axis_off()
    ax[0, 0].set_title('T+30', fontsize=24)

    #ax[0, 1].imshow(background)
    ax[0, 1].imshow(results[0][11])
    ax[0, 1].set_axis_off()
    ax[0, 1].set_title('T+60', fontsize=24)

    #ax[0, 2].imshow(background)
    ax[0, 2].imshow(results[0][17])
    ax[0, 2].set_axis_off()
    ax[0, 2].set_title('T+90', fontsize=24)

    # =================================
    #ax[1, 0].imshow(background)
    ax[1, 0].imshow(results[1][5])
    ax[1, 0].set_axis_off()

    #ax[1, 1].imshow(background)
    ax[1, 1].imshow(results[1][11])
    ax[1, 1].set_axis_off()

    #ax[1, 2].imshow(background)
    ax[1, 2].imshow(results[1][17])
    ax[1, 2].set_axis_off()

    # =================================
    #ax[2, 0].imshow(background)
    ax[2, 0].imshow(results[2][5])
    ax[2, 0].set_axis_off()

    #ax[2, 1].imshow(background)
    ax[2, 1].imshow(results[2][11])
    ax[2, 1].set_axis_off()

    #ax[2, 2].imshow(background)
    ax[2, 2].imshow(results[2][17])
    ax[2, 2].set_axis_off()
    # =================================
    #ax[3, 0].imshow(background)
    ax[3, 0].imshow(results[3][5])
    ax[3, 0].set_axis_off()

    #ax[3, 1].imshow(background)
    ax[3, 1].imshow(results[3][11])
    ax[3, 1].set_axis_off()

    #ax[3, 2].imshow(background)
    ax[3, 2].imshow(results[3][17])
    ax[3, 2].set_axis_off()
    # =================================
    #ax[4, 0].imshow(background)
    ax[4, 0].imshow(results[4][5])
    ax[4, 0].set_axis_off()

    #ax[4, 1].imshow(background)
    ax[4, 1].imshow(results[4][11])
    ax[4, 1].set_axis_off()

    #ax[4, 2].imshow(background)
    ax[4, 2].imshow(results[4][17])
    ax[4, 2].set_axis_off()
    # =================================
    #ax[5, 0].imshow(background)
    ax[5, 0].imshow(results[5][5])
    ax[5, 0].set_axis_off()

    #ax[5, 1].imshow(background)
    ax[5, 1].imshow(results[5][11])
    ax[5, 1].set_axis_off()

    #ax[5, 2].imshow(background)
    ax[5, 2].imshow(results[5][17])
    ax[5, 2].set_axis_off()

    fig.savefig('./img_ens/' + name + '.png')


def image_predict(results_list, name):
    #input [(18,384,352)]
    deter_img(results_list[0], name)
    ens_img(results_list, name)


dates_list = ['2022-11-08 16:50:00',
              '2022-10-23 20:25:00',
              '2022-09-26 19:05:00',
              '2022-10-01 00:30:00',
              '2022-09-30 19:25:00',
              '2022-09-30 21:05:00',
              '2022-09-27 18:35:00',
              '2022-09-26 16:05:00',
              '2022-09-08 20:20:00',
              '2022-09-09 14:45:00',
              '2022-09-08 17:30:00',
              '2022-07-20 22:55:00',
              '2022-07-21 02:25:00',
              '2022-07-21 04:55:00',
              '2022-06-05 17:20:00',
              '2022-06-05 20:40:00',
              '2022-06-06 14:50:00',
              '2022-06-05 22:40:00',
              '2022-06-08 09:25:00',
              '2022-06-18 22:50:00',
              '2022-05-19 12:50:00'
              ]

ROOT = get_project_root()

CHECKPOINT_DIR = ROOT / 'Checkpoints' / \
    (str("Vanilla_prec&mot2prec") + '_v' + str(0))


disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
gen_optimizer = Adam(learning_rate=5E-5, beta_1=0.0, beta_2=0.999)
loss_hinge_gen = Loss_hing_gen()
loss_hinge_disc = Loss_hing_disc()
my_model = DGMR(lead_time=90, time_delta=5)
my_model.compile(gen_optimizer, disc_optimizer,
                 loss_hinge_gen, loss_hinge_disc)

test_data = Dataset(Path('Data/test'), batch_size=1,
                    prob=False, shuffle=False)  # , shuffle=True


ckpt = tf.train.Checkpoint(generator=my_model.generator_obj,
                           discriminator=my_model.discriminator_obj,
                           generator_optimizer=my_model.gen_optimizer,
                           discriminator_optimizer=my_model.disc_optimizer)

ckpt_manager = tf.train.CheckpointManager(
    ckpt, CHECKPOINT_DIR, max_to_keep=10)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    #ckpt.restore(ckpt_manager.checkpoints[-30])
    print('Latest checkpoint restored!!')

test_data = iter(test_data)
cont = True
for cond, targ, _,date in test_data:
    if str(read_date(date)) in dates_list:

        tf.print("found date")
        results = [my_model.generator_obj(
            cond[..., :], is_training=True) for _ in range(6)]
            
        #results = [result * 0.9 for result in results]
        tf.print(results[0].shape)
        results = [tf.concat((cond[..., :1], r), axis=1) for r in results]
        tf.print(results[0].shape)
        #results = [crop_middle(r) for r in results]
        results_stacked = tf.stack(results, axis=0)
        tf.print(results_stacked.shape)
        results_stacked = results_stacked.numpy()
        #results_stacked[results_stacked < 1.] = np.nan
        tf.print(results_stacked.shape)
        results_stacked = results_stacked[:, 0, ..., 0]
        results_unstacked = tf.unstack(results_stacked, axis=0)
    
        tf.print(results_unstacked[0].shape)
        image_predict(results_unstacked, date_to_name(read_date(date)))
        video_predict(results_unstacked, date_to_name(read_date(date)))
