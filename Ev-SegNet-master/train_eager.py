import platform
import subprocess

import numpy as np
import tensorflow as tf
import os
import nets.Network as Segception
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, init_model, get_metrics
import argparse
import matplotlib.pyplot as plt 
from time import time
from tqdm import tqdm
import pickle


# Change depending on os
if platform.system() == 'Windows':
    import utils.Loader_win as Loader
else:
    import utils.Loader as Loader

# enable eager mode
# In TF 2 eager execution is enabled by default!!
# tf.enable_eager_execution()
tf.random.set_seed(7)
np.random.seed(7)


# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=None, lr=None, init_lr=2e-4, variables_to_optimize=None, evaluation=True, preprocess_mode=None, lr_pow=0.9):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0

    print("Number of epochs: " + str(epochs) + "\n")
    loss_arr = []
    
    for epoch in tqdm(range(epochs-last_epoch), desc="Epochs"):  # for each epoch
        lr_decay(lr, init_lr, 1e-9, last_epoch + epoch, epochs, power=lr_pow)  # compute the new lr
        print('epoch: ' + str(epoch + last_epoch) + '. Learning rate: ' + str(lr.numpy()))
        for step in tqdm(range(steps_per_epoch), desc="Steps per Epoch"):  # for every batch
            with tf.GradientTape() as g:
                # get batch
                x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter)

                x = preprocess(x, mode=preprocess_mode)
                [x, y, mask] = convert_to_tensors([x, y, mask])

                y_, aux_y_ = model(x, training=True, aux_loss=True)  # get output of the model

                # Rewrite https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/softmax_cross_entropy
                #loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                #loss = loss_fn(y_true=y, y_pred=y_, sample_weight=mask)  # compute loss
                #loss_aux = loss_fn(y_true=y, y_pred=aux_y_, sample_weight=mask)
                loss = tf.compat.v1.losses.softmax_cross_entropy(y, y_, weights=mask)  # compute loss
                loss_aux = tf.compat.v1.losses.softmax_cross_entropy(y, aux_y_, weights=mask)  # compute loss
                loss = 1*loss + 0.8*loss_aux
                if show_loss:
                    print('Training loss: ' + str(loss.numpy()))
            
            loss_arr.append(loss.numpy())
            # Gets gradients and applies them
            grads = g.gradient(loss, variables_to_optimize)
            optimizer.apply_gradients(zip(grads, variables_to_optimize))

        if evaluation:
            # get metrics
            #train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode)
            test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                              scales=[1], preprocess_mode=preprocess_mode)

            #print('Train accuracy: ' + str(train_acc.numpy()))
            #print('Train miou: ' + str(train_miou))
            print('Test accuracy: ' + str(test_acc.numpy()))
            print('Test miou: ' + str(test_miou))
            print('')

            # save model if better
            if test_miou > best_miou:
                best_miou = test_miou
            # Log the results
            with summary_writer.as_default():
                tf.summary.scalar('Training loss ', loss.numpy(), step=epoch+last_epoch)
                tf.summary.scalar('Test accuracy ', test_acc.numpy(), step=epoch+last_epoch)
                tf.summary.scalar('Test mIoU ', test_miou, step=epoch+last_epoch)
            
            # Try to make the saved model generally useful
            # model.save_weights(name_best_model + "model" + str(epoch + last_epoch), save_format='tf')
            # print("Written savedmodel in tf to " + name_best_model + "model" + str(epoch + last_epoch))
        else:
            pass
            # model.save_weights(name_best_model + "model" + str(epoch), save_format='tf')
            # print("Written savedmodel in tf to " + name_best_model + "model" + str(epoch + last_epoch))
        # if platform.system() != "Windows":
        #     subprocess.run(["zip", "-r", "/content/drive/MyDrive/Universiteit/Deep_Learning/logs.zip", "/content/DL_EvSegNet/Ev-SegNet-master/logs"])
        #     subprocess.run(["zip", "-r", "/content/drive/MyDrive/Universiteit/Deep_Learning/model.zip", "/content/DL_EvSegNet/Ev-SegNet-master/weights/model"])
        
        loader.suffle_segmentation()  # shuffle training set
        
    with open('results/loss_'+str(batch_size)+"_"+str(init_lr), 'wb') as f:
        pickle.dump(loss_arr, f)

def plot_param_grid(param1, param2, miou_values): 
  """
  plots the MIOU values for the hyperparameter ranges 
  Input: - param1: range of hyperparameter 1 (np.arange)
         - param2: range of hyperparameter 2 (np.arange) 
         - miou_values: matrix with miou values correspoding to param1 and param2
         combinations 
  Outputs: - heat map with MIoU vs param1 and param2
  """
  x = param1.copy() 
  y = param2.copy()
  X, Y = np.meshgrid(x, y)

  fig = plt.figure(frameon=False) 

  im1 = plt.imshow(miou_values, cmap=plt.cm.gray)

  # fix axis ticks 
  nx = x.shape[0]
  ny = y.shape[0]
  no_labels_x = len(x) # number of x labels 
  no_labels_y = len(y) # number of y labels 
  step_x = int(nx/(no_labels_x - 1)) # label step size 
  step_y = int(ny/(no_labels_y - 1)) 
  x_positions = np.arange(0, nx, step_x) # pixel count at label position 
  y_positions = np.arange(0, ny, step_y) 
  x_labels = x[::step_x]
  y_labels = y[::step_y]  
  plt.xticks(x_positions, x_labels)
  plt.yticks(y_positions, y_labels)

  # add color bar 
  # plt.colorbar(plt.pcolor(miou_values))

  plt.show()

if __name__ == "__main__":
    # Calculate time taken for data load.
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path", default='data')
    parser.add_argument("--model_path", help="Model path", default='weights/model')
    parser.add_argument("--n_classes", help="number of classes to classify", default=6)
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("--epochs", help="number of epochs to train", default=500)
    parser.add_argument("--width", help="number of epochs to train", default=352)
    parser.add_argument("--height", help="number of epochs to train", default=224)
    parser.add_argument("--lr", help="init learning rate", default=1e-3)
    parser.add_argument("--n_gpu", help="number of the gpu", default=0)
    parser.add_argument("--percentage_data_used", help="portion of the dataset used, e.g. 0.5 is 50%", default=1.0)
    parser.add_argument("--log_dir", help="where the tensorboard logs are stored", default='logs/')
    parser.add_argument("--hyperparam", help="perform hyperparameter tuning? True/False", default=0)
    parser.add_argument("--batch_size_range", help="range of batch size for hyperparameter tuning: min max step", nargs='+',)
    parser.add_argument("--lr_range", help="range of initial learning rates for hyperparameter tuning: min max step", nargs='+')
    parser.add_argument("--lr_power", help="exponent used for lr decay", default=0.9)

    # WIP
    parser.add_argument("--check_class_ratio", help="check ratio of classes after shrinking data set", choices=('True','False'), default='False')

    args = parser.parse_args()

    n_gpu = int(args.n_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    # Suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    n_classes = int(args.n_classes)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    width = int(args.width)
    height = int(args.height)
    lr = float(args.lr)
    percentage_data_used = float(args.percentage_data_used)
    check_class_ratio = args.check_class_ratio == 'True'
    hyperparam_tuning = int(args.hyperparam)
    lr_pow = float(args.lr_power)

    channels = 6  # input of 6 channels
    channels_image = 0
    channels_events = channels - channels_image
    folder_best_model = args.model_path
    folder_logs = args.log_dir
    name_best_model = os.path.join(folder_best_model, 'myBestmodel')
    dataset_path = args.dataset
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation', width=width,
                           height=height, channels=channels_image, channels_events=channels_events, percentage_data_used=percentage_data_used, check_class_ratio=check_class_ratio)

    data_load_time = time()
    print("Data has loaded in ", (data_load_time-start), 'seconds')

    if not os.path.exists(folder_best_model):
        os.makedirs(folder_best_model)
    if not os.path.exists(folder_logs):
        os.makedirs(folder_logs)

    # build model and optimizer
    model = Segception.Segception_small(num_classes=n_classes, weights=None, input_shape=(None, None, channels))
    print("SUCCESS: Build model and optimizer")

    if hyperparam_tuning == 1:
      last_epoch = 0

      # batch_size_range = list(args.batch_size_range)
      # lr_range = list(args.lr_range)

      #batch_range = np.array([4, 8]) # 2, 4, 8, 16, 32?
      # lr_range = lr_range = np.array([0.1, 0.05, 0.01, 0.005, 0.0001, 0.0005, 0.00001, 0.000005, 0.000001])
      #lr_range = lr_range = np.array([0.0005])
      lr_range = np.array([0.0005])
      batch_range = np.array([4, 8, 16])
    
      # initialise arrays containg test metrics
      test_accs = np.zeros((len(lr_range), len(batch_range)))
      mious = np.zeros((len(lr_range), len(batch_range)))

      for index_lr, learning_rate in tqdm(enumerate(lr_range)):
          print('evaluating learning rate ', learning_rate)
          # initialize learning rate
          lr = tf.Variable(learning_rate)

          # construct optimizer
          optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

          # loop over batch sizes
          for index_batch, batch_size in enumerate(batch_range):
              print('evaluating batch_size: ', batch_size)

              # build model and optimizer
              model = Segception.Segception_small(num_classes=n_classes, weights=None, input_shape=(None, None, channels))

              batch_size = int(batch_size)

              vars_to_optimize = model.variables

              # Init model
              model.build(input_shape=(batch_size, width, height, channels))

              # train model
              train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=lr,
                    init_lr=learning_rate, variables_to_optimize=vars_to_optimize, evaluation=False, preprocess_mode=None, lr_pow=lr_pow)

              test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True,
                                                scales=[1, 0.75, 1.5],
                                                write_images=False, preprocess_mode=None)
              test_accs[index_lr, index_batch] = test_acc
              mious[index_lr, index_batch] = test_miou
                
              # Store data  
              res_dict = {'acc' : test_acc.numpy(), 'miou' : test_miou}
        
              with open('results/result_'+str(batch_size)+"_"+str(learning_rate), 'wb') as f:
                  pickle.dump(res_dict, f)
        
      print(f"Lr range: {lr_range}")
      print(f"Bacthes: {batch_range}")
      print(f"Mious: {mious}")  
      # plot results
      plot_param_grid(lr_range, batch_range, mious)

      # find best performing parameters
      index_max_test_acc = np.unravel_index(np.argmax(test_accs, axis=None), test_accs.shape)
      # index_max_test_acc = np.argmax(test_accs)
      index_max_miou = np.unravel_index(np.argmax(mious, axis=None), mious.shape)
      #index_max_miou = np.argmax(mious)

      #TODO: use miou or test_acc as evaluation metric for choosing hyperparameters? Chose MIoU for now
      print('index_max_miou: ', index_max_miou)
      print('lr_range:, ', lr_range)
      best_lr = lr_range[index_max_miou[0]]
      best_batch_size = batch_range[index_max_miou[1]]
      print('MIoUs: ', mious)
      print('tes_accs: ', test_accs)
      print('Best batch size: ', best_batch_size)
      print('Best lr: ', best_lr)
      #print('MIoU of ', max(mious), ' obtained with batch size of ', best_batch_size, ' and learning rate of ', best_lr)

    else:
        # optimizer
        learning_rate = tf.Variable(lr)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        variables_to_optimize = model.variables

        # Initialize writer for tensorboard
        summary_writer = tf.summary.create_file_writer(folder_logs)

        # Init models (definitely not optional, needed to initialize buttfuck everything if you want to load the model)
        model.build(input_shape=(batch_size, width, height, channels))
        print("SUCCESS: Initialized Model")

        # restore if model saved and show number of params
        get_params(model)
        model.summary()

        # If you want to load the model before training, e.g. restore a checkpoint of a session with less than 500 epochs,
        # uncomment the following lines
        try:
            latest = tf.train.latest_checkpoint(folder_best_model)
            last_epoch = 0 #int(latest.split("myBestmodel")[1]) + 1
            model.load_weights(latest)
            print("Model " + str(last_epoch) + "loaded")
        except Exception as e:
            last_epoch = 0
            print("Last model could not be found; starting from scratch")

        train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=learning_rate,
              init_lr=lr, variables_to_optimize=variables_to_optimize, evaluation=True, preprocess_mode=None)

        # Test best model
        print('Testing model')
        model.summary()

        #test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1, 0.75, 1.5], write_images=True, preprocess_mode=None)
        test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1, 0.75, 1.5], write_images=True, preprocess_mode=None, n_samples_max=100)
        print('Test accuracy: ' + str(test_acc.numpy()))
        print('Test miou: ' + str(test_miou))
