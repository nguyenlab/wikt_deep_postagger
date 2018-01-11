# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"


import unittest
import cPickle
from saf.constants import annotation
from saf.importers import CoNLLImporter

from wikt_morphodecomp.annotation import MorphoAnalysisAnnotator


from tdv_postagger.tagging import train_model
from tdv_postagger.config.loader import load_config


UD_TRAIN_FILEPATH = "./data/UD_English/en-ud-train.conllu"
UD_TEST_FILEPATH = "./data/UD_English/en-ud-dev.conllu"
CONFIG_FILEPATH = "./data/config/default.config.json"
W2M_CONFIG_FILEPATH = "./data/config/w2m/wikt_morphodecomp.config.json"


class TestUDPOSTagger(unittest.TestCase):
    def test_training(self):
        config = load_config(CONFIG_FILEPATH)
        model = None

        print "Loading inputs..."
        try:
            with open("./data/ud_train_morpho_annot.pickle", "rb") as morphoannotated_file:
                ud_train_doc = cPickle.load(morphoannotated_file)
        except IOError:
            conll_importer = CoNLLImporter(field_list=[annotation.LEMMA, annotation.UPOS, annotation.XPOS,
                                                       annotation.FEATS, annotation.HEAD, annotation.DEPREL])
            with open(UD_TRAIN_FILEPATH) as ud_train_file:
                ud_train_doc = conll_importer.import_document(unicode(ud_train_file.read(), encoding="utf8").strip())

            print "Running morphological decomposition..."
            morpho_annotator = MorphoAnalysisAnnotator()
            morpho_annotator.annotate(ud_train_doc, ensemble=True, config_paths=(W2M_CONFIG_FILEPATH,))

            with open("./data/ud_train_morpho_annot.pickle", "wb") as morphoannotated_file:
                cPickle.dump(ud_train_doc, morphoannotated_file, cPickle.HIGHEST_PROTOCOL)

        print "Training POS tagger..."
        #ud_train_doc.sentences = ud_train_doc.sentences[0:20]

        for i in xrange(1):
            model = train_model(ud_train_doc, config, model_seq=i)

        pass


    # def test_loading(self):
    #     conll_importer = CoNLLImporter(field_list=[annotation.LEMMA, annotation.UPOS, annotation.XPOS, annotation.FEATS,
    #                                                annotation.HEAD, annotation.DEPREL])
    #
    #     with open(UD_TRAIN_FILEPATH) as ud_train_file:
    #         ud_train_doc = conll_importer.import_document(unicode(ud_train_file.read(), encoding="utf8").strip())
    #
    #     with open(UD_TEST_FILEPATH) as ud_test_file:
    #         ud_test_doc = conll_importer.import_document(unicode(ud_test_file.read(), encoding="utf8").strip())



