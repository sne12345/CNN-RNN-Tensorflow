# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import Batch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
import h5py

## GPU 사용시
## GPU 메모리 사용 크기만 할당
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
import argparse

def parser_arg():
    parser = argparse.ArgumentParser()
    ## 
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='seed number. if 0, do not fix seed (default: 0)')

    ## hyper-parameters
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help="epoch (default: 200)")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help="batch size (default: 128)")
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer (default: adam)")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help="learning rate (default: 0.01)")
    parser.add_argument('--reg', type=float, default=1e-4, help="the weight of l2 regularizer (default: 1e-4)")
    parser.add_argument('--patience', type=int, default=0, help="the patience of Early Stopping. if 0, do not use EarlyStoping (default: 0)")


    ## debug
    args, _ = parser.parse_known_args('--epochs 10 --batch_size 32 --optimizer adam --lr 0.01 --patience 10'.split())
    
    # args, _ = parser.parse_known_args()

    args.model_name = f'{args.optimizer}_lr{args.lr}_bs{args.batch_size}'
    if args.reg: args.model_name += f'_l2reg{args.reg}'
    
    return args

def load_images_from_h5py(path1, path2):
    h5f = h5py.File(path1, 'r')
    data1 = h5f.get('images')[()]
    h5f.close()
    
    h5f = h5py.File(path2, 'r')
    data2 = h5f.get('images')[()]
    h5f.close()
    
    return data1, data2

def conv(x, filters, ksize=3, strides=1, padding='same', activation='relu', pooling=True, reg=0.0):
    regulerlizer = tf.keras.regularizers.l2(reg) if reg else None
    x = Conv2D(filters=filters, kernel_size=ksize, strides=strides, 
            padding=padding, kernel_regularizer=regulerlizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=filters, kernel_size=ksize, strides=strides, 
            padding=padding, kernel_regularizer=regulerlizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if pooling:
        x = MaxPool2D((2, 2))(x)

    return x

if __name__ == '__main__':
    args = parser_arg()
    print(args)

    DATA_PATH = 'data/'
    file_list = os.listdir(DATA_PATH)

    df_Train = pd.read_csv(DATA_PATH + 'Train.csv')
    df_Test = pd.read_csv(DATA_PATH + 'Test.csv')

    image_height = 32
    image_width = 32
    image_channel = 3 # 컬러 이미지이기에 3채널

    path_train_data = 'data/train_images.h5'
    path_test_data = 'data/test_images.h5'

    train_images, test_images = load_images_from_h5py(path_train_data, path_test_data)

    # Train : 60% Valid: 40% 나누기
    train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, df_Train.ClassId, test_size=0.1)

    datagen_kwargs = dict(rescale=1./255)
    dataflow_kwargs = dict(batch_size=args.batch_size)

    # Train
    train_datagen = ImageDataGenerator(
            zoom_range=0.2,
            rotation_range=20,
            #  width_shift_range=0.2,
            #  height_shift_range=0.2,
            #  brightness_range=[0.25,1.0],
        **datagen_kwargs
    )

    train_generator = train_datagen.flow(
        train_images, 
        y=train_labels,
        **dataflow_kwargs
    )

    # Validation
    valid_datagen = ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow(
        valid_images,
        y=valid_labels,
        shuffle=False, 
        **dataflow_kwargs
    )

    # Test
    test_datagen = ImageDataGenerator(**datagen_kwargs)
    test_generator = test_datagen.flow(
        test_images, 
        y=df_Test.ClassId, 
        shuffle=False, 
        **dataflow_kwargs
    )

    """## 3. 딥러닝 모델
    ### 3-1. CNN 모델 설정
    CNN을 사용하여 간단하게 모델을 구현해 보겠습니다. filters, kernel 등의 사이즈는 하이퍼 파리미터로 자신만의 모델로 튜닝이 가능합니다.
    """

    inputs = Input((image_height, image_width, image_channel))
    x = conv(inputs, filters=32, reg=args.reg)
    x = conv(x, filters=64, reg=args.reg)
    x = conv(x, filters=128, pooling=False, reg=args.reg)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(43, activation='softmax')(x)

    model = tf.keras.Model(inputs, output)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=f'model_{args.model_name}.h5', monitor='val_accuracy', save_best_only=True),
    ]

    if args.patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=args.patience))

    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    else:
        raise NotImplementedError(f"the optimizer {args.optimizer} is not implemented")

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    # EPOCHS에 따른 성능을 보기 위하여 history 사용
    history = model.fit(
        train_generator,
        validation_data = valid_generator, # validation 데이터 사용
        epochs=args.epochs, 
        callbacks=callbacks,
    )

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(len(history.history['accuracy']))
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig('graph.png')

    model.load_weights('model.h5')
    val_loss, val_accuracy = model.evaluate(valid_generator)
    test_loss, test_accuracy = model.evaluate(test_generator)

    print('valid set accuracy: ', val_accuracy)
    print('test set accuracy: ', test_accuracy)

    # plt.figure(figsize = (13, 13))
    # x_test, y_test = test_generator.next()
    # pred_test = model.predict_on_batch(x_test)
    # pred_class = np.argmax(pred_test, axis=-1)

    # start_index = 0
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.grid(False)
    #     plt.xticks([])
    #     plt.yticks([])
    #     prediction = pred_class[start_index + i]
    #     actual = int(y_test[start_index + i])
    #     col = 'g'
    #     if prediction != actual:
    #         col = 'r'
    #     plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
    #     plt.imshow(array_to_img(x_test[start_index + i]))
    # plt.show()

    test_prediction = model.predict(test_generator)
    predicted_class = np.argmax(test_prediction, axis=-1)

    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    # cm = confusion_matrix(test_generator.y, predicted_class)
    # plt.figure(figsize = (20, 20))
    # sns.heatmap(cm, annot = True)
    # plt.show()
    print(classification_report(test_generator.y, predicted_class))