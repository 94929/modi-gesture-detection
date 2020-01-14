
import modi

import time

import numpy as np
import pandas as pd 


def record_motion(btn, gyro, train=False):
    print('recording')

    print('ready')
    time.sleep(1)
    print('start')

    l = []
    X = None
    X_df = None
    while True:
        time.sleep(0.01)

        aX = gyro.acceleration_x()
        aY = gyro.acceleration_y()
        aZ = gyro.acceleration_z()
        gX = gyro.angular_vel_x()
        gY = gyro.angular_vel_y()
        gZ = gyro.angular_vel_z()
        l.append((aX,aY,aZ,gX,gY,gZ))
        print('aX, aY, aZ, gX, gY, gZ', aX, aY, aZ, gX, gY, gZ)

        if btn.clicked():
            X = np.array(l)[5:-5]
            if len(X) > 50:
                X_tr = X[::2][:25]
            else:
                raise ValueError('data is too short')

            if train:
                filename = 'data/5.csv'
                with open(filename, 'a') as f:
                    f.write('\n')
                X_df = pd.DataFrame(X_tr, columns=['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'])
                X_df.to_csv(filename, mode='a', header=False)
            break
    return X_df

if __name__ == '__main__':
    bundle = modi.MODI()
    gyro = bundle.gyros[0]
    btn = bundle.buttons[0]
#    gyro = None
#    btn = None
#    while gyro is None or btn is None:
#        try:
#        except IndexError:
#            print('connecting to gyro and btn')
#
    df = record_motion(btn, gyro, train=True)

    tensor = []
    inputs = []
    SAMPLES_PER_GESTURE = 25
    for j in range(SAMPLES_PER_GESTURE):
        index = j
        tensor += [
            (df['aX'][index]), (df['aY'][index]), (df['aZ'][index]),
            (df['gX'][index]), (df['gY'][index]), (df['gZ'][index])
        ]
    inputs.append(tensor)
    inputs = np.array(inputs)
    print(inputs.shape)

    ## predict
    #import tensorflow as tf 
    #print(f"TensorFlow version = {tf.__version__}\n")
    #
    #model = tf.keras.models.load_model('model.h5')
    #preds = model.predict(inputs)
    #print("prediction = left: O, right: X", np.round(preds, decimals=3))

