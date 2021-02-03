#!/usr/bin/env python
# coding: utf-8

import datetime
import networkx as nx
import nltk
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytz
import re
import sys

from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from textwrap import fill, wrap

STOP_WORDS = set(nltk.corpus.stopwords.words('portuguese') + list(punctuation))
STOP_WORDS.update(['corona', 'covid', '19']) # search terms

DATA_DIR = os.path.expanduser('~/Dados/corona')
SITE_DIR = os.path.expanduser('~/Dados/covid_viz')
BULLETIN_DIR = os.path.join(SITE_DIR, 'boletins')

TIMEZONE = 'America/Sao_Paulo'

def get_files_from_date(dir_, term, date, ext='csv'):
    ls = os.listdir(DATA_DIR)

    files = []
    for i in range(7):
        file_name = '{}_{}.{}'.format(term, date.strftime('%y%m%d'), ext)
        if os.path.exists(os.path.join(dir_, file_name)):
            files.append(file_name)
        date += datetime.timedelta(days=1)

    return files

def remove_url(string):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', string)

def wrap_text(text, wrap_width=100, sep='\n'):
    return sep.join(wrap(text, wrap_width))

def plot_line(x, y, **kwargs):
    fig = go.Figure(go.Scatter(x=x, y=y))

    fig.update_layout(
        xaxis_tickformat='%d/%m/%y',
        yaxis_rangemode='tozero',
        yaxis_ticksuffix=' mil',
        hovermode='x',
        **kwargs
    )

    return fig

def plot_bar(x, y, **kwargs):
    fig = go.Figure(go.Bar(x=x, y=y))

    fig.update_layout(
        hovermode='x',
        **kwargs
    )

    return fig

def bag_of_words(series, max_features=10):
    series = series.apply(remove_url)
    vet = CountVectorizer(
        max_features=max_features,
        stop_words=STOP_WORDS
    )
    bow = vet.fit_transform(series)

    return bow, vet.get_feature_names()

def spmatrix_to_df(matrix, columns, index):
    return pd.DataFrame.sparse.from_spmatrix(matrix, columns=columns, index=index)

def plot_graph(corpus, node_size_factor=30, n_nodes=15, **kwargs):
    X, features = bag_of_words(corpus, max_features=n_nodes)
    adj_matrix = X.T * X
    adj_matrix.setdiag(0)
    adj_matrix = adj_matrix / adj_matrix.max()
    graph_df = spmatrix_to_df(adj_matrix, features, features)
    g = nx.convert_matrix.from_pandas_adjacency(graph_df)

    pos = nx.spring_layout(g)
    xn, yn = np.array(list(pos.values())).T
    xe = []
    ye = []
    for e in g.edges:
        xe += [pos[e[0]][0], pos[e[1]][0], None]
        ye += [pos[e[0]][1], pos[e[1]][1], None]

    node_size = adj_matrix.sum(axis=1) * node_size_factor
    axis = {'showline': True, 'zeroline': False, 'showgrid': False, 'showticklabels': False, 'title': ''}

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xn, y=yn,
                mode='markers+text', marker_size=node_size,
                text=list(g.nodes), textposition="middle center",
                hoverinfo='none', #TODO: number of records
            ),
            go.Scatter(x=xe, y=ye, mode='lines', line_color='rgba(151, 151, 151, 0.2)', hoverinfo='none')
        ],
        layout=go.Layout(xaxis=axis, yaxis=axis, showlegend=False, **kwargs)
    )

    return fig

def export_html(df, figures):
    dates = df['date'].sort_values().iloc[[0, -1]].to_numpy()
    file_name = '{}.html'.format(dates[0].strftime('%y%m%d'))

    if dates[0].day != dates[1].day:
        periodo = '{} a {}'.format(dates[0].strftime('%d/%m/%Y'), dates[1].strftime('%d/%m/%Y'))
    else:
        periodo = '{}'.format(dates[0].strftime('%d/%m/%Y'))

    with open(os.path.join(BULLETIN_DIR, file_name), 'w') as f:
        f.write('<h1>Covid-19 Twitter {}</h1>'.format(periodo))
        f.write('<h2>Total de tweets coletados: {}</h2>'.format(df.shape[0]))
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    with open(os.path.join(SITE_DIR, 'index.html'), 'w') as f:
        f.write('<h1>Lista de boletins da Covid-19 no Twitter</h1>')

        for file_ in np.sort(os.listdir(BULLETIN_DIR)):
            with open(os.path.join(BULLETIN_DIR, file_)) as b:
                period = b.read().split('<h1>Covid-19 Twitter ')[1].split('</h1>')[0]
                f.write('<a href="boletins/{}">{}</a><br>'.format(file_, period))

def load_data_generate_html(week_dates):
    for week_date in week_dates:
        if type(week_date) == str:
            date = datetime.datetime.strptime(week_date, '%d/%m/%Y')

        else:
            date = week_date

        date = date.replace(tzinfo=pytz.timezone(TIMEZONE))
        if date.weekday() != 6: # not a sunday
            date -= datetime.timedelta(days=(date.weekday() + 1)) # sunday of that week

        files = get_files_from_date(DATA_DIR, 'corona', date)
        files = [os.path.join(DATA_DIR, f) for f in files]
        if not len(files):
            continue

        df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)

        df.rename(
            columns={
                'user.screen_name': 'user',
                'retweeted_status.user.screen_name': 'retweeted_user',
                'complete_text': 'text'
            },
            inplace=True
        )

        df = df[df['lang'] == 'pt']

        df['date'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
        df['date'] = df['date'].dt.tz_convert(TIMEZONE)
        df = df[df['date'] >= date].reset_index(drop=True)

        date_series = df.groupby(df['date'].dt.date)['text'].count()
        tweet_dia = plot_line(
            date_series.index,
            date_series.values / 1000,
            title='Tweets por dia',
            xaxis_title='Dia',
            yaxis_title='Quantidade de Tweets',
        )

        top_tweet = df['text'].value_counts(ascending=False).index[0]
        top_tweet_series = df[df['text'] == top_tweet].groupby(df['date'].dt.date)['lang'].count()[:10]
        percurso = plot_line(
            top_tweet_series.index,
            top_tweet_series.values / 1000,
            title='Percurso top tweet',
            xaxis_title='Dia<br>Tweet: "{}"'.format(wrap_text(top_tweet, 100, '<br>')),
            yaxis_title='Quantidade de Retweets',
        )

        text_series = df['text'].value_counts(ascending=False)[:10]
        tweets_populares = plot_bar(
            [wrap_text(text, 50, '<br>') for text in text_series.index],
            text_series.values / 1000,
            title='Tweets mais populares',
            xaxis_title='Tweet',
            xaxis_showticklabels=False,
            yaxis_title='Quantidade de registros',
            yaxis_ticksuffix=' mil',
        )

        rt_user_series = df['retweeted_user'].dropna().value_counts(ascending=False)[:10]
        perfis_retweetados = plot_bar(
            rt_user_series.index,
            rt_user_series.values / 1000,
            title='Perfis mais retweetados',
            xaxis_title='Perfis',
            yaxis_title='Quantidade de Retweets',
            yaxis_ticksuffix=' mil',
        )

        user_series = df['user'].dropna().value_counts(ascending=False)[:10]
        perfis_ativos = plot_bar(
            user_series.index,
            user_series.values,
            title='Perfis mais ativos',
            xaxis_title='Perfis',
            yaxis_title='Quantidade de registros',
        )

        text_series = df['text'].dropna()
        grafo = plot_graph(text_series, title='Conexão de palavras')

        export_html(df, [
            tweet_dia, tweets_populares, percurso, perfis_retweetados,
            perfis_ativos, grafo
        ])

if len(sys.argv) == 2:
    date = sys.argv[1]
else:
    # generate previous week graphs
    now = datetime.datetime.now()
    date = now - datetime.timedelta(days=(now.weekday() + 2))

load_data_generate_html([date])
