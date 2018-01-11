#-*- coding: utf8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import numpy as np
import json
import string

CHAR_CODES = dict([(unicode(c), idx) for (idx, c) in enumerate(list(string.ascii_lowercase) + ['\'', '-', '{', '}', ' ', chr(1)])])
ENC_SIZE_CHAR = len(CHAR_CODES)

pos_db = dict()


def encode_char(c, char_size=ENC_SIZE_CHAR):
    enc = np.zeros(char_size, dtype=np.uint8)

    if (c.lower() in CHAR_CODES):
        c_code = CHAR_CODES[c.lower()]
        enc[c_code] = 1
    else:
        enc[CHAR_CODES[unichr(1)]] = 1

    return enc


def encode_token(w, word_size, char_size, char_win_size, reverse=False):
    enc = np.zeros((word_size, char_size * char_win_size), dtype=np.uint8)

    assert (char_win_size % 2) == 1
    assert char_win_size >= 1

    if (len(w) > word_size):
        print "Warning: Token exceeds maximum length: %d. Will be truncated." % word_size
        print w
        print "Length: ", len(w)

    w = w[0:word_size - 2]

    if (not reverse):
        charseq = list("{"+ w + "}")
    else:
        charseq = list(u"{"+ w[::-1] + u"}")

    lpadded = char_win_size // 2 * [u"{"] + charseq + char_win_size // 2 * [u"}"]
    context_windows = [lpadded[i:(i + char_win_size)] for i in range(len(charseq))]

    for i in xrange(len(context_windows)):
        enc[i] = np.concatenate([encode_char(c) for c in context_windows[i]])

    return enc


def encode_attributes(token, attr_dim, max_morphemes, posdb_path, pos_idx):
    global pos_db
    assert attr_dim >= len(pos_idx)

    enc = np.zeros((max_morphemes + 1, attr_dim), dtype=np.float16)

    if (not pos_db):
        with open(posdb_path) as posdb_file:
            pos_db = json.load(posdb_file)

    attr_idx = 0
    if ("MORPHO" in token.annotations):
        for morpheme in token.annotations["MORPHO"]["decomp"]:
            if (morpheme in pos_db and attr_idx < max_morphemes):
                i = 1
                for pos in pos_db[morpheme]:
                    if (pos in pos_idx):
                        enc[attr_idx][pos_idx[pos]] = 1.0 / i
                    i += 1

                attr_idx += 1

        #while (attr_idx < max_morphemes):
        #    enc[attr_idx][pos_idx[":end:"]] = 1.0
        #    attr_idx += 1

    if (token.surface in pos_db):
        i = 1
        for pos in pos_db[token.surface]:
            if (pos in pos_idx):
                enc[max_morphemes][pos_idx[pos]] = 1.0 / i
            i += 1

    return enc.flatten()


def encode_sentence(sent, max_sent_len, token_size, ctx_win_size, char_win_size, attr_dim, max_morphemes, posdb_path, pos_idx):
    enc_token_windows = np.zeros((max_sent_len, token_size * ctx_win_size, ENC_SIZE_CHAR * char_win_size), dtype=np.uint8)
    enc_attr_windows = np.zeros((max_sent_len, attr_dim * (max_morphemes + 1) * ctx_win_size), dtype=np.float16)

    enc_tokens = []
    enc_attrs = []

    if (len(sent.tokens) > max_sent_len):
        print "Warning: Sentence exceeds maximum length: %d. Will be truncated." % max_sent_len
        print [tok.surface for tok in sent.tokens]
        print "Length: ", len(sent.tokens)

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        token = sent.tokens[i]
        if ("MORPHO" in token.annotations):
            enc_tokens.append(encode_token(" ".join(token.annotations["MORPHO"]["decomp"]), token_size, ENC_SIZE_CHAR,
                                           char_win_size))
        else:
            enc_tokens.append(encode_token(token.surface, token_size, ENC_SIZE_CHAR, char_win_size))
            print "Warning: Tokens without morphological annotations. Using surface forms only."

        enc_attrs.append(encode_attributes(token, attr_dim, max_morphemes, posdb_path, pos_idx))

    start_token = np.zeros((token_size, ENC_SIZE_CHAR * char_win_size), dtype=np.uint8)
    end_token = np.zeros((token_size, ENC_SIZE_CHAR * char_win_size), dtype=np.uint8)
    for i in xrange(token_size):
        start_token[i] = np.concatenate([encode_char(u"{")] * char_win_size)
        end_token[i] = np.concatenate([encode_char(u"}")] * char_win_size)

    start_attr = np.zeros(attr_dim, dtype=np.uint8)
    start_attr[pos_idx[":start:"]] = 1
    start_attr = np.concatenate([start_attr] * (max_morphemes + 1))
    end_attr = np.zeros(attr_dim, dtype=np.uint8)
    end_attr[pos_idx[":end:"]] = 1
    end_attr = np.concatenate([end_attr] * (max_morphemes + 1))

    padded_token_seq = ctx_win_size // 2 * [start_token] + enc_tokens + ctx_win_size // 2 * [end_token]
    padded_attr_seq = ctx_win_size // 2 * [start_attr] + enc_attrs + ctx_win_size // 2 * [end_attr]

    padded_token_windows = [padded_token_seq[i:(i + ctx_win_size)] for i in range(len(enc_tokens))]
    padded_attr_windows = [padded_attr_seq[i:(i + ctx_win_size)] for i in range(len(enc_attrs))]

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        enc_token_windows[i] = np.concatenate(padded_token_windows[i], axis=0)
        enc_attr_windows[i] = np.concatenate(padded_attr_windows[i], axis=0)

    return (enc_token_windows, enc_attr_windows)


def encode_pos(sent, max_sent_len, pos_map, pos_idx):
    enc = np.zeros((max_sent_len, 1), dtype=np.uint8)

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        tagged_classes = pos_map.get(sent.tokens[i].annotations["UPOS"], "x").split("|")
        enc[i] = pos_idx[tagged_classes[0]]
    for i in xrange(len(sent.tokens), max_sent_len):
        enc[i] = pos_idx[":end:"]

    return enc


def encode_document(doc, config, training=False):
    max_sent_len = config["model"]["MAX_SENT_LEN"]
    token_size = config["model"]["ENC_SIZE_TOKEN"]
    char_win_size = config["model"]["ENC_CHAR_SIZE_CTX_WIN"]
    ctx_win_size = config["model"]["ENC_SIZE_CTX_WIN"]
    num_morphemes = config["model"]["MAX_MORPHEMES"]
    posdb_path = config["data"]["posdb_path"]
    pos_idx = dict([(pos, idx) for (idx, pos) in list(enumerate(sorted(config["model"]["POS_CLASSES"])))])
    pos_map = config["model"]["POS_MAP"]

    sent_encs = []
    pos_encs = []
    sample_weight_encs = []

    for sentence in doc.sentences:
        sent_encs.append(encode_sentence(sentence, max_sent_len, token_size, ctx_win_size, char_win_size,
                                         len(pos_idx), num_morphemes, posdb_path, pos_idx))

        if (training):
            pos_encs.append(encode_pos(sentence, max_sent_len, pos_map, pos_idx))
            sample_weight_encs.append(encode_sample_weights(sentence, max_sent_len))

    return sent_encs, pos_encs, sample_weight_encs


def encode_sample_weights(sent, max_sent_len):
    enc = np.zeros(max_sent_len, dtype=np.uint8)

    for i in xrange(len(sent.tokens[0:max_sent_len])):
        enc[i] = 1

    return enc
