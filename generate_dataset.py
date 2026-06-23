import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_boxing_dataset():
    data = []

    classes = {
        'Ekselente': 38,
        'Diak':      52,
        'Moderadu':  38,
        'Fraku':     22
    }

    ranges = {
        'Ekselente': {
            'vo2max':        (55.0, 70.0),
            'pushup':        (55,   75),
            'situp':         (55,   72),
            'shuttle_run':   (9.0,  10.2),
            'sprint_30m':    (3.8,  4.2),
            'sit_reach':     (38.0, 52.0),
            'body_fat':      (6.0,  12.0),
            'bmi':           (20.0, 23.0),
            'grip_strength': (55.0, 72.0),
            'vertical_jump': (65.0, 82.0),
        },
        'Diak': {
            'vo2max':        (45.0, 55.0),
            'pushup':        (40,   56),
            'situp':         (40,   56),
            'shuttle_run':   (10.2, 11.5),
            'sprint_30m':    (4.2,  4.7),
            'sit_reach':     (28.0, 38.0),
            'body_fat':      (12.0, 16.0),
            'bmi':           (22.0, 25.0),
            'grip_strength': (45.0, 56.0),
            'vertical_jump': (52.0, 66.0),
        },
        'Moderadu': {
            'vo2max':        (35.0, 45.0),
            'pushup':        (25,   41),
            'situp':         (25,   41),
            'shuttle_run':   (11.5, 12.8),
            'sprint_30m':    (4.7,  5.2),
            'sit_reach':     (16.0, 28.0),
            'body_fat':      (16.0, 20.0),
            'bmi':           (24.0, 27.0),
            'grip_strength': (32.0, 46.0),
            'vertical_jump': (38.0, 53.0),
        },
        'Fraku': {
            'vo2max':        (20.0, 35.0),
            'pushup':        (10,   26),
            'situp':         (10,   26),
            'shuttle_run':   (12.8, 15.0),
            'sprint_30m':    (5.2,  6.5),
            'sit_reach':     (2.0,  16.0),
            'body_fat':      (20.0, 28.0),
            'bmi':           (26.0, 32.0),
            'grip_strength': (15.0, 32.0),
            'vertical_jump': (18.0, 39.0),
        },
    }

    for label, count in classes.items():
        r = ranges[label]
        for _ in range(count):
            row = {
                'vo2max':        round(np.random.uniform(*r['vo2max']), 2),
                'pushup':        int(np.random.randint(*r['pushup'])),
                'situp':         int(np.random.randint(*r['situp'])),
                'shuttle_run':   round(np.random.uniform(*r['shuttle_run']), 2),
                'sprint_30m':    round(np.random.uniform(*r['sprint_30m']), 2),
                'sit_reach':     round(np.random.uniform(*r['sit_reach']), 1),
                'body_fat':      round(np.random.uniform(*r['body_fat']), 1),
                'bmi':           round(np.random.uniform(*r['bmi']), 1),
                'grip_strength': round(np.random.uniform(*r['grip_strength']), 1),
                'vertical_jump': round(np.random.uniform(*r['vertical_jump']), 1),
                'prontidaun_fiziku': label,
            }
            data.append(row)

    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, 'id_atleta', [f'ATL{str(i+1).zfill(3)}' for i in range(len(df))])
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_boxing_dataset()
    df.to_csv('data/boxing_dataset.csv', index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(df['prontidaun_fiziku'].value_counts())
    print(df.head(10).to_string())
