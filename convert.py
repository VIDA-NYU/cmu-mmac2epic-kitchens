import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tqdm
from bs4 import BeautifulSoup

COL_ORDER = 'narration_id,participant_id,video_id,narration_timestamp,start_timestamp,stop_timestamp,start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes'
COL_ORDER = COL_ORDER.split(',')

REMOVE_SUFFIXES = ['tl', 'tr', 'tm', 'bl', 'br', 'bm', 'l', 's', 'm']

NON_NOUNS = ['1', '2', 'partailly', 'partially', 'patially']  # a bit manual but

def main(ann_dir='Data', out_dir='output', target_recipe=None, suffix='_retrieval_test', make_sentence_csv=True, **kw):
    # load src csv files
    dfs = []
    for f in tqdm.tqdm(glob.glob(os.path.join(ann_dir, 'Plans/*/*.txt'))):
        df = convert_csv(f, target_recipe=target_recipe, **kw)
        if df is None:
            continue
        dfs.append(df)
    df = pd.concat(dfs)
    df, _, _ = verb_noun_classes(df)

    # set column/row ordering
    df = df[COL_ORDER]
    df = df.set_index('narration_id').sort_index()

    # csv output prefix
    _suffix = f'_{target_recipe}' if target_recipe else ''
    name = f'cmu_mmac{_suffix}{suffix}'
    print(name)

    # write out csvs/pkl files
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'{name}.csv'))
    df.to_pickle(os.path.join(out_dir, f'{name}.pkl'))

    if make_sentence_csv:
        sentence_df = df[['narration']].reset_index().groupby('narration').first().reset_index().set_index('narration_id')
        sentence_df.to_csv(os.path.join(out_dir, f'{name}_sentence.csv'))
        sentence_df.to_pickle(os.path.join(out_dir, f'{name}_sentence.pkl'))
    
    print(df)


def verb_noun_classes(df):
    # get unique noun/verb
    unique_verbs = df.verb.unique()
    unique_nouns = np.unique(['']+[n for ns in df.all_nouns for n in ns])
    print('verb classes:', len(unique_verbs), sorted(unique_verbs))
    print('noun classes:', len(unique_nouns), sorted(unique_nouns))

    # convert noun/verb to index
    verb_i_lookup = {v: i for i, v in enumerate(unique_verbs)}
    noun_i_lookup = {n: i for i, n in enumerate(unique_nouns)}
    df['verb_class']  = df.verb.apply(lambda v: verb_i_lookup[v])
    df['noun_class']  = df.noun.apply(lambda n: noun_i_lookup[n])
    df['all_noun_classes']  = df.all_nouns.apply(lambda ns: [noun_i_lookup[n] for n in ns])
    return df, unique_verbs, unique_nouns


def convert_csv(csv_file, eaf_file=None, fps=30, target_recipe=None, norm=True):
    # read csv
    eaf_file = os.path.splitext(csv_file)[0] + '.eaf'
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['start_time', 'new', 'narration']
    df['stop_time'] = df.start_time.shift(-1)

    # remove parentheses
    df['narration'] = df.narration.str.strip('()')

    # remove start end tokens
    df = df.iloc[1:-1]
    
    # set top level meta
    participant = os.path.splitext(os.path.basename(csv_file))[0]
    recipe = os.path.basename(os.path.dirname(csv_file))
    if target_recipe and recipe.lower() != target_recipe.lower():
        return
    df['participant_id'] = participant
    df['video_id'] = f'{participant}_{recipe}'
    df['narration_id'] = df.index.to_series().apply(lambda i: f'{participant}_{recipe}_{i:07d}')
    
    # get time relative to video
    soup = BeautifulSoup(open(eaf_file), 'xml')
    tstart = int(soup.find("TIME_SLOT", {"TIME_SLOT_ID":"ts1"})['TIME_VALUE'])

    # start/end timestamps
    df['start_timestamp'] = df.start_time.apply(lambda t: timedelta(seconds=(t + tstart)/1000))
    df['stop_timestamp'] = df.stop_time.apply(lambda t: timedelta(seconds=(t + tstart)/1000))
    df['narration_timestamp'] = df['start_timestamp']

    # start/end frames
    df['start_frame'] = (df.start_timestamp.dt.total_seconds() * fps).astype(int)
    df['stop_frame'] = (df.stop_timestamp.dt.total_seconds() * fps).astype(int)

    # narration to noun/verb
    df['verb'] = df.narration.apply(lambda s: s.split()[0])
    df['noun'] = df.narration.apply(lambda s: ' '.join(s.split()[1:2]))
    df['all_nouns'] = df.narration.apply(lambda s: [si for ss in s.split()[1:] for si in ss.split('&')])
    df['all_nouns'] = df.all_nouns.apply(lambda ns: [remove_suffix(n) for n in ns if n not in NON_NOUNS] or ['-'])
    df['noun'] = df.all_nouns.apply(lambda ns: ' '.join(ns[:1]))

    if norm:
        df['narration'] = df.apply(lambda r: ' '.join([r.verb]+r.all_nouns), axis=1)

    return df

def remove_suffix(x, suffixes=REMOVE_SUFFIXES, sep='_'):
    for s in suffixes:
        s = f'{sep}{s}'
        if x.endswith(s):
            return x[:-len(s)]
    return x

if __name__ == '__main__':
    import fire
    fire.Fire(main)

