__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

from keras.models import Sequential, Model, load_model
from keras.layers import Masking, TimeDistributed, LSTM, Dense, Bidirectional, Dropout, Input, concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
import keras.backend as K

import encoder as enc
from tdv_postagger.ml.crf import ChainCRF, create_custom_objects


class DeepTDVPOSTagger(object):
    def __init__(self, config):
        init_dist = config["model"]["INIT_DIST"]
        enc_char_size_ctx_win = config["model"]["ENC_CHAR_SIZE_CTX_WIN"]
        enc_size_ctx_win = config["model"]["ENC_SIZE_CTX_WIN"]

        max_sent_len = config["model"]["MAX_SENT_LEN"]
        max_morphemes = config["model"]["MAX_MORPHEMES"]
        token_size = config["model"]["ENC_SIZE_TOKEN"]
        hidden_dim_morph = config["model"]["HIDDEN_DIM_MORPH"]
        hidden_dim_dec = config["model"]["HIDDEN_DIM_DEC"]
        hidden_dim_attr = config["model"]["HIDDEN_DIM_ATTR"]
        dim_attrs = len(config["model"]["POS_CLASSES"]) * (max_morphemes + 1) * enc_size_ctx_win

        self.morpho_input = Input(shape=(None, token_size * enc_size_ctx_win, enc.ENC_SIZE_CHAR * enc_char_size_ctx_win))
        self.morpho_encoder = TimeDistributed(Masking(mask_value=0.))(self.morpho_input)
        self.morpho_encoder = TimeDistributed(
            LSTM(hidden_dim_morph[0], return_sequences=False, implementation=2, activation="tanh"),
            input_shape=(token_size * enc_size_ctx_win, enc.ENC_SIZE_CHAR * enc_char_size_ctx_win))(self.morpho_encoder)
        self.morpho_encoder = Dropout(0.2)(self.morpho_encoder)
        self.morpho_encoder = TimeDistributed(Dense(hidden_dim_morph[1], activation="tanh"))(self.morpho_encoder)

        self.attr_input = Input(shape=(None, dim_attrs))
        # self.attr_encoder = Masking(mask_value=0.)(self.attr_input)
        # self.attr_encoder.add(Bidirectional(LSTM(hidden_dim_attr[0], return_sequences=True, implementation=2, activation="tanh")))
        # self.attr_encoder.add(Dropout(0.25))
        # self.attr_encoder.add(TimeDistributed(Dense(hidden_dim_attr[1], activation="tanh")))

        self.decoder_input = concatenate([self.morpho_encoder, self.attr_input], axis=2)
        self.decoder = Bidirectional(LSTM(hidden_dim_dec[0], return_sequences=True, implementation=2, activation="tanh"))(self.decoder_input)
        self.decoder = Dropout(0.25)(self.decoder)
        self.decoder = TimeDistributed(Dense(hidden_dim_dec[1], activation="tanh"))(self.decoder)
        # self.decoder = TimeDistributed(Dense(len(config["model"]["POS_CLASSES"]), activation="softmax"))(self.decoder)
        self.decoder = TimeDistributed(Dense(len(config["model"]["POS_CLASSES"])))(self.decoder)

        crf = ChainCRF()
        self.decoder = crf(self.decoder)

        self.model = Model(inputs=[self.morpho_input, self.attr_input], outputs=[self.decoder])

        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", sample_weight_mode="temporal")
        self.model.compile(loss=crf.sparse_loss, optimizer="adam", metrics=["sparse_categorical_accuracy"], sample_weight_mode="temporal")

        self.model.summary()
        # plot_model(self.model, to_file="model.png", show_shapes=True)

    def train(self, input_seqs, output_seqs, num_epochs, batch_size=1, validation_split=0., sample_weight=None):
        self.model.fit(input_seqs, output_seqs, batch_size, num_epochs, verbose=1, validation_split=validation_split, sample_weight=sample_weight)

    def train_from_generator(self, generator, steps_per_epoch, num_epochs):
        self.model.fit_generator(generator, steps_per_epoch, epochs=num_epochs)
    
    def predict(self, insts):
        return self.model.predict(insts)

    def load(self, path):
        try:
            self.model = load_model(path, custom_objects=create_custom_objects())
            print "Loaded model."
            return True
        except(IOError):
            return False

    def save(self, path):
        self.model.save(path)


#import json
#model = DeepTDVPOSTagger(json.load(open("../data/config/default.config.json")))
