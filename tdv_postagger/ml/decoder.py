__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np


def decode_sentence(enc_sent_pos, pos_classes):
    return [pos_classes[np.argmax(enc_pos)] for enc_pos in enc_sent_pos]


def confidence(enc_sent_pos):
    return [float(np.nanmax(enc_pos)) for enc_pos in enc_sent_pos]