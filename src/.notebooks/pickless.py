#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:38:49 2019

@author: tom
"""
import numpy as np;
from datetime import datetime as dt;
import os, json;
from scipy import io;

def output(opt, d1, c1, d2, c2):
    date = "results/"+str(opt["train_amount"])+dt.now().strftime(":%Y%m%d%H%M%S");
    os.mkdir(date);
    filename = date+"/data.npz";
    np.savez(filename, d1=d1, c1=c1, d2=d2, c2=c2);
    json_file = open(date+"/option.json", 'w');
    json.dump(opt, json_file, indent=2);
    