#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Danilo S. Carvalho <danilo@jaist.ac.jp>"

import sys
import json

LANG = "English"


def main(argv):
    with open(argv[1]) as wiktdbfile:
        wiktdb = json.load(wiktdbfile)

    posdb = dict()

    for wiktentry in wiktdb:
        if (LANG in wiktentry["langs"]):
            posdb[wiktentry["title"]] = wiktentry["langs"][LANG]["pos_order"]

    with open(argv[2], "w") as posdbfile:
        json.dump(posdb, posdbfile)



if __name__ == "__main__":
    main(sys.argv)
