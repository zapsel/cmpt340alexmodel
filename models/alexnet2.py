alexnet = models.Sequential()

# 1st Convolutional Layer
alexnet.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
alexnet.add(Activation('relu'))
# Max Pooling
alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
alexnet.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
alexnet.add(Activation('relu'))
# Max Pooling
alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
alexnet.add(Activation('relu'))

# 4th Convolutional Layer
alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
alexnet.add(Activation('relu'))

# 5th Convolutional Layer
alexnet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
alexnet.add(Activation('relu'))
# Max Pooling
alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
alexnet.add(Flatten())
# 1st Fully Connected Layer
alexnet.add(Dense(4096, input_shape=(224*224*3,)))
alexnet.add(Activation('relu'))
# Add Dropout to prevent overfitting
alexnet.add(Dropout(0.4))

# 2nd Fully Connected Layer
alexnet.add(Dense(4096))
alexnet.add(Activation('relu'))
# Add Dropout
alexnet.add(Dropout(0.4))

# 3rd Fully Connected Layer
alexnet.add(Dense(1000))
alexnet.add(Activation('relu'))
# Add Dropout
alexnet.add(Dropout(0.4))

# Output Layer
alexnet.add(Dense(17))
alexnet.add(Activation('softmax'))

alexnet.summary()