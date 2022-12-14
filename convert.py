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

def main(ann_dir='Data', ek_root=None, out_dir='output', target_recipe=None, suffix='_retrieval_test', make_sentence_csv=True, **kw):
    # load src csv files
    dfs = []
    for f in tqdm.tqdm(glob.glob(os.path.join(ann_dir, 'Plans/*/*.txt'))):
        df = convert_csv(f, ek_root=ek_root, target_recipe=target_recipe, **kw)
        if df is None:
            continue
        dfs.append(df)
    df = pd.concat(dfs)
    df, _, _ = verb_noun_classes(df)

    # set column/row ordering
    df = df[COL_ORDER]
    df = df.set_index('narration_id').sort_index()

    # csv output prefix
    if target_recipe:
        suffix = f'_{target_recipe}{suffix}'
    if ek_root:
        suffix = f'_norm{suffix}'
    name = f'cmu_mmac{suffix}'
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
    print(df.narration.value_counts())


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


def convert_csv(csv_file, ek_root=None, eaf_file=None, fps=30, target_recipe=None):
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
    df['all_nouns'] = df.all_nouns.apply(lambda ns: [remove_suffix(n) for n in ns if n not in NON_NOUNS] or [])
    df['noun'] = df.all_nouns.apply(lambda ns: ' '.join(ns[:1]))
    df['narration'] = df.apply(lambda r: ' '.join([r.verb]+r.all_nouns), axis=1)

    if ek_root:
        df = norm_noun_verb(df, ek_root)
    return df


def norm_noun_verb(df, ek_root):
    noun_df = pd.read_csv(os.path.join(ek_root, 'EPIC_100_noun_classes.csv'))
    verb_df = pd.read_csv(os.path.join(ek_root, 'EPIC_100_verb_classes.csv'))

    df = df[(df.all_nouns != '-') & (df.verb != 'other')].copy()
    df['verb'] = df.verb.str.replace('_', '-')

    noun_norm = {v: row.key for i, row in noun_df.iterrows() for v in eval(row.instances)}
    verb_norm = {v: row.key for i, row in verb_df.iterrows() for v in eval(row.instances)}
    noun_norm.update(manual_noun_norm)

    def do_norm(r):
        nouns = r.all_nouns

        r['verb'] = norm_verb = verb_norm[r.verb]
        r['all_nouns'] = norm_nouns = [noun_norm[n] for n in nouns]
        r['noun'] = ' '.join(norm_nouns[:1])
        r['narration'] = ' '.join([norm_verb] + norm_nouns)
        return r

    return df.apply(do_norm, axis=1)


manual_noun_norm = {
    'measuring_cup': 'cup', 
    'counter_place': 'top', 
    'fridge_place': 'fridge', 
    'egg_box': 'box',
    'egg_shell': 'shell:egg', 
    'open_egg_shell': 'shell:egg', 
    'empty_egg_shell': 'shell:egg', 
    'oil_bottle': 'bottle',
    'butter_spray_can': 'can', 
    'oil_bottle_cap': 'cap', 
    'butter_spray_cap': 'cap', 
    'brownie_box': 'box', 
    'brownie_bag': 'bag',
    'brownie_mix': 'mixture', 
    'baking_pan': 'pan', 
    'pepper_shaker': 'cellar:salt', 
    'salt_shaker': 'cellar:salt', 
    'eggs': 'egg', 
    'bread_bag': 'bag', 
    'jam_glass': 'jar',
    'peanut_butter_glass': 'jar', 
    'peanut_butter_glass_cap': 'cap', 
    'peanut_butter': 'spreads',
    'jam_glass_cap': 'jar', 
    'paper_towel': 'towel:kitchen', 
    'fridge_counter': 'fridge', 
    'hands': 'hand',
    'jam_glass_2': 'jar',
}



def remove_suffix(x, suffixes=REMOVE_SUFFIXES, sep='_'):
    for s in suffixes:
        s = f'{sep}{s}'
        if x.endswith(s):
            return x[:-len(s)]
    return x

if __name__ == '__main__':
    import fire
    fire.Fire(main)

