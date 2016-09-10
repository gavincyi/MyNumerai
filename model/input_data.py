#!/usr/bin/env python

import pandas as pd
import os

def get_file_name(date):
    if date == '':
        return '', ''
    else:
        train = os.path.abspath(__file__ + "/../../data/%s/numerai_training_data.csv" % date)
        tour = os.path.abspath(__file__ + "/../../data/%s/numerai_tournament_data.csv" % date)
        return train, tour

def get_data(date=''):
    train_name, tour_name = get_file_name(date)
    train = pd.read_csv(train_name, delimiter=',')
    tour = pd.read_csv(tour_name, delimiter=',', index_col=[0])

    return train, tour

def output_pred(data, model, date=''):
    if date != '':
        file_name = os.path.abspath(__file__ + "/../../data/%s/%s.csv" % (date, model))
    else:
        file_name = os.path.abspath(__file__ + "/../../data/%s.csv" % model)
    data.to_csv(file_name)
