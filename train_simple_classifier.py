from get_all_signs_n_prepare import train_imdir, train_label, valid_imdir, valid_label
from network import Network, generate_processed_batch
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
#from keras.optimizers import Adam
from keras.optimizers import SGD
#%%Hyperparameters
img_rows, img_cols = 256, 256
channel = 3
num_classes = 14
batch_size = 32
nb_epoch = 50

input_shape = (img_rows, img_cols, channel)
filepath_name = 'traffic_sign_classify_model2.h5'
model = Network(input_shape,num_classes)
#%%


def scheduler(epoch):

    if epoch!=0 and epoch%10 == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.5)
        print("lr changed to {}".format(lr*.5))

    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)

callbacks_list= [
    ModelCheckpoint(
        filepath= filepath_name,
        mode='min',
        monitor='val_loss',
        save_weights_only = False,
        save_best_only=True,
        verbose = 1
    ), lr_decay]


training_gen = generate_processed_batch(train_imdir, train_label, img_size = input_shape, num_classes = num_classes, batch_size = batch_size)
val_gen = generate_processed_batch(valid_imdir,valid_label, img_size = input_shape, num_classes = num_classes, batch_size = batch_size)

#%%
sgd = SGD(lr=0.0001, momentum=0.9,  decay=7.5e-3, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(training_gen,steps_per_epoch=int(len(train_imdir)/batch_size),nb_epoch=nb_epoch,validation_data=val_gen,
                    validation_steps=int(len(valid_imdir)/batch_size),callbacks=callbacks_list,initial_epoch=0)
