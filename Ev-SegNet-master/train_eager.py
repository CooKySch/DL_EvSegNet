import numpy as np
import tensorflow as tf
import os
import nets.Network as Segception
# Change depending on os
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, init_model, get_metrics
import argparse
from time import time
from tqdm import tqdm

# enable eager mode
# In TF 2 eager execution is enabled by default!!
# tf.enable_eager_execution()
tf.random.set_seed(7)
np.random.seed(7)


# Trains the model for certains epochs on a dataset
def train(loader, model, epochs=5, batch_size=2, show_loss=False, augmenter=None, lr=None, init_lr=2e-4, variables_to_optimize=None, evaluation=True, preprocess_mode=None):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0

    print("Number of epochs: " + str(epochs) + "\n")

    for epoch in tqdm(range(epochs), desc="Epochs"):  # for each epoch
        lr_decay(lr, init_lr, 1e-9, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch) + '. Learning rate: ' + str(lr.numpy()))
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

            # Log the results to TensorBoard

            # save model if better
            if test_miou > best_miou:
                best_miou = test_miou
                # Try to make the saved model generally useful
                model.save(os.path.join(name_best_model, str(epoch)), save_format='tf')
                print("Written savedmodel in tf to " + name_best_model + str(epoch))
        else:

            model.save(os.path.join(name_best_model, str(epoch)), save_format='tf')
            print("Written savedmodel in tf to " + name_best_model + str(epoch))

        loader.suffle_segmentation()  # shuffle training set


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

    channels = 6  # input of 6 channels
    channels_image = 0
    channels_events = channels - channels_image
    folder_best_model = args.model_path
    name_best_model = os.path.join(folder_best_model, 'best')

    dataset_path = args.dataset
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width, height=height, channels=channels_image, channels_events=channels_events)

    data_load_time = time()
    print("Data has loaded in ", (data_load_time-start), 'seconds')

    if not os.path.exists(folder_best_model):
        os.makedirs(folder_best_model)

    # build model and optimizer
    model = Segception.Segception_small(num_classes=n_classes, weights=None, input_shape=(None, None, channels))

    print("SUCCESS: Build model and optimizer")

    # optimizer
    learning_rate = tf.Variable(lr)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    variables_to_optimize = model.variables

    # Init models (definitely not optional, needed to initialize buttfuck everything if you want to load the model)
    model.build(input_shape=(batch_size, width, height, channels))

    print("SUCCESS: Initialized Model")
    # restore if model saved and show number of params
    get_params(model)
    model.summary()

    train(loader=loader, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=learning_rate,
          init_lr=lr, variables_to_optimize=variables_to_optimize, evaluation=True, preprocess_mode=None)

    # Test best model
    print('Testing model')
    model.summary()
    model.load_weights(name_best_model)
    model.summary()

    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1, 0.75, 1.5],
                                      write_images=False, preprocess_mode=None)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou))