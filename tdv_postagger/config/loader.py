# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import json


def load_config(path):
    config = {}
    with open(path) as config_file:
        config = json.load(config_file)

    return config
