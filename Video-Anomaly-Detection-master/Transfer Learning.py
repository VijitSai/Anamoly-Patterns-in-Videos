# example of using a pre-trained model as a classifier
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

Xtrain, Xtest, ytrain, ytest = train_test_split(res_img2, res_label2, test_size=0.2, random_state=8)

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

vgg_model.layers
vgg_model.summary()
len(vgg_model.layers)

vgg_model.trainable = False

inp = keras.Input(shape=(227, 227, 3))

x = vgg_model(inp)

# Convert features of shape `base_model.output_shape[1:]` to vectors
#x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1000, input_shape=(7, 7, 512))(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dropout(0.7)(x)
outputs = keras.layers.Dense(2)(x)
outputs = keras.layers.Activation('sigmoid')(outputs)
model = keras.Model(inp, outputs)

my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2), 
            tf.keras.callbacks.TerminateOnNaN()]

model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(Xtrain, ytrain, batch_size=16, epochs=6, validation_split = 0.2, callbacks=[tf.keras.callbacks.TerminateOnNaN()])


model.evaluate(Xtest, ytest, verbose = 0)

print(model.history)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


### Fine tuning
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

opt = keras.optimizers.Adam(1e-5)
model.compile(
    optimizer=opt,  # Low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy'],)

model.fit(Xtrain, ytrain, epochs=10, validation_split=0.2, callbacks=[callback])
