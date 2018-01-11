__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import saf.constants as annot_const
from saf import Document
from saf.annotators import Annotator, AnnotationError
# from tagging import load_models, EnsembleMode
from tagging import pos_tag
from config import DEFAULT_CONFIG_PATH


def get_pos(doc, config_paths, ensemble):
    if (ensemble):
        pass
        # models = load_models(config_paths)
        # tagged_doc = postag_ensemble(doc, config_paths, models=models, mode=EnsembleMode.MAJORITY_OVERALL)
    else:
        tagged_doc = pos_tag(doc, config_paths[0])

    return tagged_doc


class POSAnnotator(Annotator):
    def annotate(self, annotable, ensemble=False, config_paths=(DEFAULT_CONFIG_PATH,)):
        if (annotable.__class__.__name__ == "Document"):
            return POSAnnotator.annotate_document(annotable, ensemble, config_paths)
        else:
            raise AnnotationError("This annotator only accepts document annotables.")

    @staticmethod
    def annotate_document(document, ensemble, config_paths):
        return get_pos(document, config_paths, ensemble)



