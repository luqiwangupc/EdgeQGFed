import argparse
import csv
from pathlib import Path

import numpy as np


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _read_rows(path):
    with open(path, 'r', newline='', encoding='utf-8', errors='ignore') as file:
        reader = csv.reader(file)
        rows = [row for row in reader if row]
    if rows and any(cell.lower() in {'label', 'class', 'target'} for cell in rows[0]):
        rows = rows[1:]
    return rows


def _label_index(row):
    if len(row) >= 43 and _is_float(row[-1]) and not _is_float(row[-2]):
        return len(row) - 2
    return len(row) - 1


def _encode_label(label):
    label = str(label).strip().lower()
    return 0 if label in {'0', 'normal', 'normal.', 'benign'} else 1


def _ignored_columns(row, label_col):
    ignored = {label_col}
    if label_col == len(row) - 2 and _is_float(row[-1]):
        ignored.add(len(row) - 1)
    return ignored


def _build_features(rows, label_col):
    feature_rows = []
    for row in rows:
        ignored = _ignored_columns(row, label_col)
        feature_rows.append([value for idx, value in enumerate(row) if idx not in ignored])

    num_cols = len(feature_rows[0])
    categorical_values = {}
    numeric_cols = []
    for col in range(num_cols):
        values = [row[col] for row in feature_rows]
        if all(_is_float(value) for value in values):
            numeric_cols.append(col)
        else:
            categorical_values[col] = sorted(set(values))

    encoded_rows = []
    for row in feature_rows:
        encoded = []
        for col in numeric_cols:
            encoded.append(float(row[col]))
        for col, values in categorical_values.items():
            value_to_index = {value: idx for idx, value in enumerate(values)}
            one_hot = [0.0] * len(values)
            one_hot[value_to_index[row[col]]] = 1.0
            encoded.extend(one_hot)
        encoded_rows.append(encoded)

    return np.asarray(encoded_rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Prepare network-flow CSV/TXT data as EdgeQGFed npz.')
    parser.add_argument('--input', required=True, help='Path to raw NSL-KDD/flow CSV or TXT file.')
    parser.add_argument('--output', default='data/nslkdd/dataset.npz', help='Output npz path.')
    parser.add_argument('--users-column', type=int, default=-1, help='Optional user/host column index. Default: disabled.')
    args = parser.parse_args()

    rows = _read_rows(args.input)
    if not rows:
        raise ValueError(f'No rows found in {args.input}')

    label_col = _label_index(rows[0])
    X = _build_features(rows, label_col)
    y = np.asarray([_encode_label(row[label_col]) for row in rows], dtype=np.int64)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.users_column >= 0:
        users = np.asarray([row[args.users_column] for row in rows])
        np.savez_compressed(output_path, X=X, y=y, users=users)
    else:
        np.savez_compressed(output_path, X=X, y=y)

    print(f'Saved {output_path} with X={X.shape}, y={y.shape}, classes={sorted(np.unique(y).tolist())}')


if __name__ == '__main__':
    main()
