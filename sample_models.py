from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, pooling=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    
    return (output_length + stride - 1) // (stride * pooling)

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_layer = input_data
    for i in range(recur_layers):
        rnn_layer = GRU(units, activation='relu', return_sequences=True, implementation=2, name=f'deep_rnn_{i+1}')(rnn_layer)
        rnn_layer = BatchNormalization()(rnn_layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation="relu",
        return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, 
                units, recur_layers, conv_layers, dropout=0.3, max_pool=1, output_dim=29):
    """ Build a deep network for speech 
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # TODO: Specify the layers in your network
    
    cnn_layer = input_data
    for i in range(conv_layers):
        # Add convolutional layer(s)
        cnn_layer = Conv1D(filters, kernel_size, 
                           strides=conv_stride, 
                           padding=conv_border_mode,
                           activation='relu',
                           name=f'conv1d_{i+1}')(cnn_layer)

        # Add (cnn) batch normalization
        cnn_layer = BatchNormalization(name=f'bn_conv1d_{i+1}')(cnn_layer)

        # Add (cnn) dropout
        cnn_layer = Dropout(dropout, name=f'cnn_dropout_{i+1}')(cnn_layer)
        
    # Add max pooling 
    #pool_layer = MaxPooling1D(max_pool, name=f'cnn_max_pool_{i+1}')(cnn_layer)
    
    # Add recurrent layers, each with batch normalization
    rnn_layer = cnn_layer
    for i in range(recur_layers):
        rnn_layer = Bidirectional(GRU(units, activation='relu', return_sequences=True, 
                                      implementation=2, dropout_W=0.3, dropout_U=0.0, name=f'deep_rnn_{i+1}'))(rnn_layer)
        rnn_layer = BatchNormalization()(rnn_layer)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense_A = TimeDistributed(Dense(output_dim))(rnn_layer)
    
    # Dropout A
    dropout_A = Dropout(dropout) (time_dense_A)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    #time_dense_B = TimeDistributed(Dense(output_dim))(dropout_A)
    
    # Dropout B
    #dropout_B = Dropout(dropout) (time_dense_B)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(dropout_A)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    
    print(model.summary())
    return model