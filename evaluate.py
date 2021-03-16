import os
import re
from glob import glob
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from docopt import docopt
from grakel.datasets import fetch_dataset
from calc_kernel import Kernel

__doc__ = '''Evaluate performance of the kernel.
Usage:
  {f} <dataset> <kernel> [--output-convergence-warning]
  {f} -h | --help
Options:
  -h --help                     Show this screen.
  --output-convergence-warning  Remove the convergence warning suppression. Kernels should be compared for performance as-is, without scaling, so the warnings are suppressed.
'''.format(f=__file__)


CANDIDATE_PARAMS = {
        'C': [10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3],
        }

FOLD_RANDOM = [66, 35, 70, 46, 85, 54, 23, 56, 91, 93]

def load_pack(directory_path, max_index):
    table_from_pack = np.zeros((max_index + 1, max_index + 1))
    packs = glob(f'{directory_path}/*.pack')
    for pack in packs:
        small_index = int(re.search('(\d+)\.pack', pack).group(1))
        with open(pack, 'r') as f:
            content = f.read().split('\n')
            for header, kernel_value in zip(content[::2], content[1::2]):
                large_index = int(re.search('-(\d+)', header).group(1))
                table_from_pack[small_index, large_index] = float(kernel_value)
    upperright_table = np.copy(table_from_pack).T
    upperright_table[range(max_index + 1), range(max_index + 1)] = 0
    table_from_pack += upperright_table
    return table_from_pack


def se(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))


def main():
    args = docopt(__doc__)

    kernel = Kernel[args['<kernel>']]
    dataset_name = args['<dataset>']

    if kernel == Kernel.RW:
        dirpath = f'random_walk-{dataset_name}'
    elif kernel == Kernel.SP:
        dirpath = f'shortest_path-{dataset_name}'
    elif kernel == Kernel.GS:
        dirpath = f'graphlet_sampling-{dataset_name}'
    elif kernel == Kernel.QK_BH_ve:
        dirpath = f'quantum_BH_ve-{dataset_name}'
    elif kernel == Kernel.QK_BH_ved:
        dirpath = f'quantum_BH_ved-{dataset_name}'
    elif kernel == Kernel.QK_SH_ve:
        dirpath = f'quantum_SH_ve-{dataset_name}'
    elif kernel == Kernel.QK_SH_ved:
        dirpath = f'quantum_SH_ved-{dataset_name}'

    if not os.path.exists(dirpath):
        assert(f'There is no {dirpath}, please run `python3 calc_kernel.py {dataset_name} {kernel.name}`.')
        return

    used_indices = np.array(sorted([int(re.search('(\d+).pack', pack).group(1)) for pack in glob(f'{dirpath}/*.pack')]))[:, np.newaxis]

    dataset = fetch_dataset(dataset_name)
    used_targets = [dataset.target[index][0] for index in used_indices]

    scoring = ('accuracy', 'f1') if np.unique(used_targets).shape[0] == 2 else ('accuracy', 'f1_macro')
    table = load_pack(dirpath, len(dataset.target) - 1)
    kernel = lambda a, b: table[np.ix_(a.flatten().astype(int), b.flatten().astype(int))]

    if not args['--output-convergence-warning']:
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning) # We want to compare the raw kernel performance without scaling.

    accs = []
    f1s = []
    for rand in FOLD_RANDOM:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rand)
        model = SVC(kernel=kernel, random_state=None, verbose=False, max_iter=10000)
        gs = GridSearchCV(estimator=model, param_grid=CANDIDATE_PARAMS, cv=skf)
        scores = cross_validate(gs, used_indices, used_targets, cv=skf, scoring=scoring, return_train_score=True)

        train_acc = scores['train_accuracy'].mean()
        train_f1 = scores['train_' + scoring[1]].mean()
        test_acc = scores['test_accuracy'].mean()
        test_f1 = scores['test_' + scoring[1]].mean()
        print(f'train_acc: {train_acc}, train_f1: {train_f1}, test_acc: {test_acc}, test_f1: {test_f1}')
        accs.append(test_acc)
        f1s.append(test_f1)

    print(f'acc: {np.mean(accs)}({np.std(accs, ddof=1)})')
    print(f'f1: {np.mean(f1s)}({np.std(f1s, ddof=1)})')


if __name__ == '__main__':
    main()
