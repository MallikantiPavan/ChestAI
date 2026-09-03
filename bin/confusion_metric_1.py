import os
import sys
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

parser = argparse.ArgumentParser(description='Compute confusion matrix')

parser.add_argument('--pred_csv_path', default='test/test.csv',
                    metavar='PRED_CSV_PATH', type=str,
                    help='Path to the prediction csv')
parser.add_argument('--true_csv_path', default='dev.csv',
                    metavar='TRUE_CSV_PATH', type=str,
                    help='Path to the ground truth csv')
parser.add_argument('--out_path', default='test/', metavar='OUT_PATH',
                    type=str, help='Path to output files')
parser.add_argument('base_name', default=None, metavar='BASE_NAME',
                    type=str, help='Base name of output files')
parser.add_argument('--prob_thred', default=0.5, type=float,
                    help='Probability threshold')


def read_csv(csv_path, true_csv=False):
    image_paths = []
    probs = []
    dict_ = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
             {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}]

    with open(csv_path) as f:
        header = f.readline().strip('\n').split(',')
        for line in f:
            fields = line.strip('\n').split(',')
            image_paths.append(fields[0])
            if not true_csv:
                probs.append(list(map(float, fields[1:])))
            else:
                prob = []
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        prob.append(dict_[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        prob.append(dict_[0].get(value))
                probs.append(list(map(int, prob)))

    probs = np.array(probs)
    return image_paths, probs, header


def get_study(path):
    return path[0:path.rfind('/')]


def transform_csv(input_path, output_path):
    infile = pd.read_csv(input_path)
    infile = infile.fillna('Unknown')
    infile.Path.str.split('/')
    infile['Study'] = infile.Path.apply(get_study)
    outfile = infile.drop('Path', axis=1).groupby('Study').max().reset_index()
    outfile.to_csv(output_path, index=False)


def transform_csv_en(input_path, output_path):
    infile = pd.read_csv(input_path)
    infile = infile.fillna('Unknown')
    infile.Path.str.split('/')
    infile['Study'] = infile.Path.apply(get_study)
    outfile = infile.drop('Path', axis=1).groupby('Study').mean().reset_index()
    groups = infile.drop('Path', axis=1).groupby('Study')
    outfile['Cardiomegaly'] = groups['Cardiomegaly'].min().reset_index()[
        'Cardiomegaly']
    outfile['Edema'] = groups['Edema'].max().reset_index()['Edema']
    outfile['Consolidation'] = groups['Consolidation'].mean().reset_index()[
        'Consolidation']
    outfile['Atelectasis'] = groups['Atelectasis'].mean().reset_index()[
        'Atelectasis']
    outfile['Pleural Effusion'] = groups['Pleural Effusion'].mean(
    ).reset_index()['Pleural Effusion']
    outfile.to_csv(output_path, index=False)


def run(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    pred_done = os.path.join(args.out_path, 'pred_csv_done.csv')
    true_done = os.path.join(args.out_path, 'true_csv_done.csv')

    transform_csv_en(args.pred_csv_path, pred_done)
    transform_csv(args.true_csv_path, true_done)

    images_pred, probs_pred, _ = read_csv(pred_done)
    images_true, probs_true, header_true = read_csv(true_done, True)

    assert images_pred == images_true

    header = [header_true[7], header_true[10], header_true[11],
              header_true[13], header_true[15]]

    rows = []
    matrices = []
    for i, label in enumerate(header):
        y_pred = probs_pred[:, i]
        y_true = probs_true[:, i]
        y_hat = (y_pred >= args.prob_thred).astype(int)

        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true, y_hat, labels=[0, 1]).ravel()
        acc = metrics.accuracy_score(y_true, y_hat, normalize=True)

        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print(label, 'tn/fp/fn/tp:', int(tn), int(fp), int(fn), int(tp),
              'acc:', float(acc), 'auc:', float(auc))

        rows.append({
            'Label': label,
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'ACC': float(acc),
            'AUC': float(auc),
            'Threshold': float(args.prob_thred)
        })
        matrices.append((label, np.array([[tn, fp], [fn, tp]])))

    out_file = os.path.join(args.out_path,
                            args.base_name + '_confusion_matrix.csv')
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print('Saved:', out_file)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes = axes.flatten()
    for ax, (label, matrix) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_title(label)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])

        for row in range(2):
            for col in range(2):
                ax.text(col, row, int(matrix[row, col]),
                        ha='center', va='center', color='black')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[-1].axis('off')
    fig.suptitle('Confusion Matrix', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    img_file = os.path.join(args.out_path,
                            args.base_name + '_confusion_matrix.png')
    fig.savefig(img_file, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', img_file)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
