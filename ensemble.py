import pandas as pd
import numpy as np


def stepwise_blend(preds, y, score_func, N=20, greater_is_better=True):

    ensemble = np.zeros_like(preds[0])
    res = pd.DataFrame(index=range(N), columns=['model', 'score'])

    for n in range(1,N+1):

        best_score = -1e+10

        for j in range(len(preds)):

            candidate = (ensemble*(n-1) + preds[j])/n
            score = score_func(y, candidate)
            if not greater_is_better:
                score = -score
            if score > best_score:
                best_candidate_idx = j
                best_candidate = candidate
                best_score = score

        ensemble = best_candidate
        if not greater_is_better:
            best_score = -best_score
        res.loc[n-1] = [best_candidate_idx, best_score]
        print('Step %d. Add %s. Score %.6f' % (n, best_candidate_idx, best_score))

    w = (res.groupby('model').size()/n).loc[range(len(preds))].fillna(0)
    print(w[w>0])

    return res, w.values
