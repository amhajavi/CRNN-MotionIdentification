from keras.layers import  Conv1D,\
                          LSTM,\
                          Dense,\
                          TimeDistributed,\
                          MaxPooling1D,\
                          Flatten,\
                          Input,\
                          Bidirectional,\
                          BatchNormalization,\
                          GRU
\
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam

def crnn(input_shape = (10, 62), representation_dim=512):
    cnn = Sequential()
    cnn.add(Conv1D(128, kernel_size=(3), strides=(1), activation='relu', padding='valid', name="Conv1" , input_shape=input_shape))
    print(cnn.output_shape)
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(256, kernel_size=(3), strides=(1), activation='relu', padding='valid', name="Conv2"))
    print(cnn.output_shape)
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(512, kernel_size=(3), strides=(1), activation='relu', padding='valid', name="Conv3"))
    print(cnn.output_shape)
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(512, kernel_size=(2), strides=(1), activation='relu', padding='valid', name="Conv4"))
    print(cnn.output_shape)
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(512, kernel_size=(2), strides=(1), activation='relu', padding='valid', name="Conv5"))
    print(cnn.output_shape)
    cnn.add(BatchNormalization())
    cnn.add(Flatten())

    model = Sequential()
    model.add(TimeDistributed(cnn, input_shape=(None, 10, 62)))
    model.add(Bidirectional(LSTM(representation_dim, return_sequences=True)))
    model.add(Bidirectional(LSTM(representation_dim, return_sequences=False, dropout=0.2)))
    model.add(Dense(144, activation='softmax'))
    model.summary()

    optimizer = RMSprop(lr=0.0001, decay=0.00001)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
