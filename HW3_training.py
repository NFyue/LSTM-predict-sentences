import os  
import numpy as np 
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import InceptionResNetV2
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Merge, Flatten, RepeatVector, TimeDistributed, Concatenate
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image as Image
from keras.preprocessing import sequence as Sequence
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model, to_categorical
from collections import Counter
from keras.layers import Bidirectional

CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# If you are running on your own computer, please change the following paths to your local paths.
# If you are running on HPC, you can keep the following paths.
# IMAGE_ROOT = '/Users/xincoder/Documents/Dataset/flickr30k_images/flickr30k_images'
# TRAIN_CAPTION_PATH = '/Users/xincoder/Documents/Dataset/flickr30k_images/train.txt'
# TEST_CAPTION_PATH = '/Users/xincoder/Documents/Dataset/flickr30k_images/test.txt'

IMAGE_ROOT = '/share/ceph/mcc7cse498/xil915/flickr30k_images/flickr30k_images'
TRAIN_CAPTION_PATH = '/share/ceph/mcc7cse498/xil915/flickr30k_images/train.txt'
TEST_CAPTION_PATH = '/share/ceph/mcc7cse498/xil915/flickr30k_images/test.txt'

WORDS_PATH = 'words.txt'
SENTENCE_MAX_LENGTH = 100 # In this dataset, the maximum length is 84.
EMBEDDING_SIZE = 256
IMAGE_SIZE = 224

CHECK_ROOT = 'checkpoint/'
if not os.path.exists(CHECK_ROOT):
    os.makedirs(CHECK_ROOT)

class Data_generator(object):
    def __init__(self, pra_batch_size=20, pra_word_frequency=2):
        self.word_frequency = pra_word_frequency # remove words whose frequency less than this value
        self.train_image_names, self.train_image_captions, self.test_image_names, self.test_image_captions = self.get_name_caption()
        self.train_image_captions_index = self.caption2index(self.train_image_captions)
        self.test_image_captions_index = self.caption2index(self.test_image_captions)
        self.batch_size = pra_batch_size # how many samples we want to train in each step
        self.train_steps_epoch = len(self.train_image_names)//pra_batch_size # steps per epoch
        self.test_steps_epoch = len(self.test_image_names)//pra_batch_size # steps per epoch
        
    def get_name_caption(self):
        '''
        Load training and testing data from files. 
        We add a <SOS> and <EOS> to the beginning and the end of each sentence respectively.
        (<SOS> stands for "start of sentence", <EOS> stands for "end of sentence")
        Returns:
            train_image_name_list: all paths of training images
            train_caption_list: corresponding training captions
            test_image_name_list: all paths of testing images
            test_caption_list: corresponding testing captions
        '''
        with open(TRAIN_CAPTION_PATH, 'r') as reader:
            content = [x.strip().split('\t') for x in reader.readlines()]
            train_image_name_list = [os.path.join(IMAGE_ROOT, x[0].split('#')[0]) for x in content]
            train_caption_list = ['<SOS> {} <EOS>'.format(x[1].lower()) for x in content]

        with open(TEST_CAPTION_PATH, 'r') as reader:
            content = [x.strip().split('\t') for x in reader.readlines()]
            test_image_name_list = [os.path.join(IMAGE_ROOT, x[0].split('#')[0]) for x in content]
            test_caption_list = ['<SOS> {} <EOS>'.format(x[1].lower()) for x in content]    

        all_words = ' '.join(train_caption_list+test_caption_list).split(' ')
        words_num = Counter(all_words)
        words = [x for x in words_num if words_num[x]>=self.word_frequency]
        print('{} unique words (all).'.format(len(words_num)))
        print('{} unique words (count>={}).'.format(len(words), self.word_frequency))

        with open(WORDS_PATH, 'w') as writer:
            writer.write('\n'.join(words))

        return train_image_name_list, train_caption_list, test_image_name_list, test_caption_list

    def get_dictionary(self, pra_captions):
        ''' 
        Generate a dictionary for all words in our dataset. 
        Return:
            words2index: word->index dictionary 
            index2words: index->word dictionary
        '''
        if not os.path.exists(WORDS_PATH):
            words = set(' '.join(pra_captions).split(' '))
            with open(WORDS_PATH, 'w') as writer:
                writer.write('\n'.join(words))
        else:
            with open(WORDS_PATH, 'r') as reader:
                words = [x.strip() for x in reader.readlines()]

        self.voc_size = len(words)
        words2index = dict((w, ind) for ind, w in enumerate(words, start=0))
        index2words = dict((ind, w) for ind, w in enumerate(words, start=0))
        return words2index, index2words

    def caption2index(self, pra_captions):
        words2index, index2words = self.get_dictionary(pra_captions)
        captions = [x.split(' ') for x in pra_captions]
        index_captions = [[words2index[w] for w in cap if w in words2index.keys()] for cap in captions]
        return index_captions

    def index2caption(self, pra_index):
        words2index, index2words = self.get_dictionary('')
        captions = [' '.join([index2words[w] for w in cap]) for cap in pra_index]
        return captions 

    def convert2onehot(self, pra_caption):
        captions = np.zeros((len(pra_caption), self.voc_size))
        for ind, cap in enumerate(pra_caption, start=0):
            captions[ind, cap] = 1
        return np.array(captions)

    def get_epoch_steps(self):
        return self.train_steps_epoch, self.test_steps_epoch

    def generate(self, pra_train=True):
        '''
        This is a generator which is used to continuously generate training or testing data.
            pra_train = True : generate training data
            pra_train = False : generate testing data
        '''
        while True:
            if pra_train:
                # we shuffle training data at the beginning of each epoch.
                shuffle_index = np.random.permutation(len(self.train_image_names))
                image_name_list = np.array(self.train_image_names)[shuffle_index]
                image_caption_list = np.array(self.train_image_captions)[shuffle_index]
                image_caption_index_list = np.array(self.train_image_captions_index)[shuffle_index]
            else:
                image_name_list = self.test_image_names
                image_caption_list = self.test_image_captions
                image_caption_index_list = self.test_image_captions_index

            image_caption_index_list = Sequence.pad_sequences(image_caption_index_list, maxlen=SENTENCE_MAX_LENGTH, padding='post')
            
            input_image_list = []
            input_caption_list = []
            target_caption_list = []
            for index, (image_name, image_caption) in enumerate(zip(image_name_list, image_caption_index_list), start=1):
                # image
                input_image = Image.img_to_array(Image.load_img(image_name, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3)))
                input_caption_onehot = self.convert2onehot(image_caption)
                target_caption_onehot = np.zeros_like(input_caption_onehot)
                target_caption_onehot[:-1] = input_caption_onehot[1:]
                
                input_image_list.append(input_image)
                input_caption_list.append(input_caption_onehot)
                target_caption_list.append(target_caption_onehot)

                if len(input_image_list) == self.batch_size:
                    tmp_images = np.array(input_image_list)
                    tmp_captions = np.array(input_caption_list)
                    tmp_targets = np.array(target_caption_list)
                    input_image_list = []
                    input_caption_list = []
                    target_caption_list = []
                    yield [preprocess_input(tmp_images), tmp_captions], tmp_targets


class Image_Caption(object):
    def __init__(self, pra_voc_size):
        self.voc_size = pra_voc_size

        # Model design start from here.
        # we use the VGG16 as the base model to extract CNN feature from an image
        base_model = VGG16(weights='imagenet', include_top=True)
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

        '''
        !!!!!!!!!!!!!!change base_model to InceptionResNetV2!!!!!!!!!!!
        # '''
        # base_model = InceptionResNetV2(include_top=True, weights='imagenet')
        # base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('Dropout').output)

        for layer in base_model.layers[1:]:
            layer.trainable = False

        # add a fully connected layer on the top of our base model 
        # and repeat it several times, so that it has the same shape as our language model
        image_model = Sequential()
        image_model.add(base_model)
        image_model.add(Dense(EMBEDDING_SIZE, activation='relu'))
        image_model.add(RepeatVector(SENTENCE_MAX_LENGTH))
        
        # we use an Embedding layer to generate a good representation for captions.
        language_model = Sequential()
        # language_model.add(Embedding(self.voc_size, EMBEDDING_SIZE, input_length=SENTENCE_MAX_LENGTH))
        language_model.add(LSTM(128, input_shape=(SENTENCE_MAX_LENGTH, self.voc_size), return_sequences=True))
        

        language_model.add(TimeDistributed(Dense(128)))


        model = Sequential()
        model.add(Merge([image_model, language_model], mode='concat'))
        # model.add(Concatenate([image_model, language_model]))
        model.add(LSTM(1000, return_sequences=True))

        '''
        !!!!!!!!!!!!bidirectional LSTM!!!!!!!!!
        '''
        model.add(Bidirectional(LSTM(1000, return_sequences=True)))

        # !!!!!!!!!!!increase the number of LSTM layer!!!!!!!!!!
        # model.add(LSTM(1000, return_sequences=True))

        # model.add(Dense(self.voc_size, activation='softmax', name='final_output'))
        model.add(TimeDistributed(Dense(self.voc_size, activation='softmax')))

        
        self.model = model
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def train_model(self, pra_datagen):
        # callback: draw curve on TensorBoard
        tensorboard = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
        # callback: save the weight with the highest validation accuracy
        filepath=os.path.join(CHECK_ROOT, 'weights-improvement-{val_acc:.4f}-{epoch:04d}.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')

        # train model 
        '''!!!!!!!!!!!change epoch from 2 to 5!!!!!!!!!!'''
        self.model.fit_generator(
            pra_datagen.generate(True), 
            steps_per_epoch=pra_datagen.get_epoch_steps()[0], 
            epochs=5, 
            validation_data=pra_datagen.generate(False), 
            validation_steps=pra_datagen.get_epoch_steps()[1],
            callbacks=[tensorboard, checkpoint])
    
if __name__ == '__main__':
    my_generator = Data_generator()
    model = Image_Caption(my_generator.voc_size)
    model.train_model(my_generator)