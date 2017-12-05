import os  
import cv2
import numpy as np 

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.models import Sequential, Model

from keras.layers import LSTM, Dense, Embedding, Merge, Flatten, RepeatVector, TimeDistributed, Concatenate

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing import image as Image

from keras.preprocessing import sequence as Sequence

from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.utils import plot_model, to_categorical

from collections import Counter


CUDA_VISIBLE_DEVICES='0'

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


WORDS_PATH = 'words.txt'

SENTENCE_MAX_LENGTH = 100 # In this dataset, the maximum length is 84.

EMBEDDING_SIZE = 256

IMAGE_SIZE = 224


CHECK_ROOT = 'checkpoint/'

if not os.path.exists(CHECK_ROOT):

    os.makedirs(CHECK_ROOT)

with open(WORDS_PATH, 'r') as reader:
    words = [x.strip() for x in reader.readlines()]


def get_dictionary(pra_captions):

    # print words[5101]
    voc_size = len(words)

    words2index = dict((w, ind) for ind, w in enumerate(words, start=0))

    index2words = dict((ind, w) for ind, w in enumerate(words, start=0))

    return words2index, index2words


def caption2index(pra_captions):

    words2index, index2words = get_dictionary(pra_captions)

    captions = [x.split(' ') for x in pra_captions]

    index_captions = [[words2index[w] for w in cap if w in words2index.keys()] for cap in captions]

    return index_captions


def index2caption(pra_index):

    words2index, index2words = get_dictionary('')

    captions = [' '.join([index2words[w] for w in cap]) for cap in pra_index]

    return captions 


def convert2onehot(index):

    onehot = np.zeros((1,100, 12503))
    # a = pra_caption[0][:][:]
    for ind, cap in enumerate(index[0], start=0):

        onehot[0,ind, cap[0]] = 1

    return np.array(onehot)


# class Image_Caption(object):
global voc_size 
voc_size = 12503
#     def __init__(self, pra_voc_size):


base_model = VGG16(weights='imagenet', include_top=True)

base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

for layer in base_model.layers[1:]:

    layer.trainable = False


image_model = Sequential()

image_model.add(base_model)

image_model.add(Dense(EMBEDDING_SIZE, activation='relu'))

image_model.add(RepeatVector(SENTENCE_MAX_LENGTH))


language_model = Sequential()



language_model.add(LSTM(128, input_shape=(SENTENCE_MAX_LENGTH, voc_size), return_sequences=True))

language_model.add(TimeDistributed(Dense(128)))



model = Sequential()

model.add(Merge([image_model, language_model], mode='concat'))

# model.add(Concatenate([image_model, language_model]))

model.add(LSTM(1000, return_sequences=True))

# model.add(Dense(self.voc_size, activation='softmax', name='final_output'))

model.add(TimeDistributed(Dense(voc_size, activation='softmax')))


# draw the model and save it to a file.

# plot_model(model, to_file='model.pdf', show_shapes=True)



model = model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.load_weights('/Users/yueshu/Desktop/flickr30k_images/xin_weights.hdf5')

captions = ['<SOS>']


# img = cv2.imread('/Users/yueshu/Desktop/RNN/image/6905083.jpg')

# resized_img = cv2.resize(img, (224,224)).astype('float32')

# img0 = resized_img.reshape(1, 224,224,3).astype('float32')

input_image = Image.img_to_array(Image.load_img('/Users/yueshu/Desktop/RNN/image/65567.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE, 3)))
# img =  cv2.imread('/Users/yueshu/Desktop/RNN/image/6905083.jpg')
# resized_img = cv2.resize(img, (224,224)).astype('float32')
image = np.array([input_image])



i = 0
while(i < 100):
    print 'i'
    print i
    # if(i == 0):
    #     captions.append('<SOS>')
    if(captions[i] == '<EOS>'):
        print captions
        break
    else:
        ind = caption2index(captions)
        print ind
        indPad = Sequence.pad_sequences([ind], maxlen=100, padding='post')
        # print 'indPad'
        # print indPad

        one = convert2onehot(indPad)

        # one = Sequence.pad_sequences(one)
        # for ii in range(0,100):
        #     if one[0][ii][0] == 1.0:
        #         # print ii
        #         one[0][ii][0] = 0.0
        print one

        predict = model.predict_classes({'input_1':preprocess_input(image),'lstm_1_input':one})
        print predict
        pre = predict[0][i]
        # print pre
        xx = words[pre]
        captions.append(xx)        
        print 'captions'
        print captions
        i += 1
        # ind = caption2index(captions)
        # print ind
        # indx = ind[i][0]
        # indPad[0][i] = indx
captions = ' '.join(captions) 
print captions






