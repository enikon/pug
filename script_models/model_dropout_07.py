import tensorflow as tf


def create_model(classes_number, split_counts):
    #                    # batch size, num time steps, num features
    inputs_ch = tf.keras.Input(shape=(split_counts[0], 1))
    inputs_cd = tf.keras.Input(shape=(split_counts[1], 1))
    inputs_cw = tf.keras.Input(shape=(split_counts[2], 1))
    inputs_th = tf.keras.Input(shape=(split_counts[3],))
    inputs_tw = tf.keras.Input(shape=(split_counts[4],))
    inputs_tc = tf.keras.Input(shape=(split_counts[5],))

    # Double CNN initial on sequential data

    cyclic_params_first = {
        'filters': 5,
        'kernel_size': 3,
        'activation': 'relu',
        'padding': 'same'
    }
    cyclic_params_second = {
        'filters': 15,
        'kernel_size': 3,
        'activation': 'relu',
        'padding': 'same',
    }

    cnn1h = tf.keras.layers.Conv1D(
        **cyclic_params_first
    )(inputs_ch)
    cnn1d = tf.keras.layers.Conv1D(
        **cyclic_params_first
    )(inputs_cd)
    cnn1w = tf.keras.layers.Conv1D(
        **cyclic_params_first
    )(inputs_cw)

    cnn2h = tf.keras.layers.Conv1D(
        **cyclic_params_second
    )(cnn1h)
    cnn2d = tf.keras.layers.Conv1D(
        **cyclic_params_second
    )(cnn1d)
    cnn2w = tf.keras.layers.Conv1D(
        **cyclic_params_second
    )(cnn1w)

    #mp1h = tf.keras.layers.MaxPool1D()(cnn2h)
    #mp1d = tf.keras.layers.MaxPool1D()(cnn2d)
    #mp1w = tf.keras.layers.MaxPool1D()(cnn2w)

    bn1cnn1h = tf.keras.layers.BatchNormalization()(cnn2h)
    bn1cnn1d = tf.keras.layers.BatchNormalization()(cnn2d)
    bn1cnn1w = tf.keras.layers.BatchNormalization()(cnn2w)

    # RNN on sequential data

    lstm1h = tf.keras.layers.LSTM(units=5)(bn1cnn1h)
    lstm1d = tf.keras.layers.LSTM(units=5)(bn1cnn1d)
    lstm1w = tf.keras.layers.LSTM(units=5)(bn1cnn1w)

    bn1lstm1h = tf.keras.layers.BatchNormalization()(lstm1h)
    bn1lstm1d = tf.keras.layers.BatchNormalization()(lstm1d)
    bn1lstm1w = tf.keras.layers.BatchNormalization()(lstm1w)

    # Concatenate all
    concat0 = tf.keras.layers.Concatenate()([
        bn1lstm1h,
        bn1lstm1d,
        bn1lstm1w,
        inputs_th,
        inputs_tw,
        inputs_tc
    ])

    # DNN Categorical
    dense1 = tf.keras.layers.Dense(35, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(concat0)
    dense2 = tf.keras.layers.Dense(20, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(dense1)
    dp2 = tf.keras.layers.Dropout(0.15)(dense2)

    if classes_number == 0:
        outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(dp2)
    elif classes_number == 2:
        outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dp2)
    else:
        outputs = tf.keras.layers.Dense(classes_number, activation=tf.nn.softmax)(dp2)

    model = tf.keras.Model(
        inputs=[
            inputs_ch,
            inputs_cd,
            inputs_cw,
            inputs_th,
            inputs_tw,
            inputs_tc
        ], outputs=outputs)

    return model

