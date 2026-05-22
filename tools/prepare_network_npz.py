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


NSLKDD_ATTACK_GROUPS = {
    'normal': 0,
    'normal.': 0,
    'dos': 1,
    'back': 1,
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'apache2': 1,
    'udpstorm': 1,
    'processtable': 1,
    'mailbomb': 1,
    'worm': 1,
    'probe': 2,
    'satan': 2,
    'ipsweep': 2,
    'nmap': 2,
    'portsweep': 2,
    'mscan': 2,
    'saint': 2,
    'r2l': 3,
    'guess_passwd': 3,
    'ftp_write': 3,
    'imap': 3,
    'phf': 3,
    'multihop': 3,
    'warezmaster': 3,
    'warezclient': 3,
    'spy': 3,
    'xlock': 3,
    'xsnoop': 3,
    'snmpguess': 3,
    'snmpgetattack': 3,
    'httptunnel': 3,
    'sendmail': 3,
    'named': 3,
    'u2r': 4,
    'buffer_overflow': 4,
    'loadmodule': 4,
    'rootkit': 4,
    'perl': 4,
    'sqlattack': 4,
    'xterm': 4,
    'ps': 4,
}


def _encode_label(label):
    label = str(label).strip().lower()
    if label in NSLKDD_ATTACK_GROUPS:
        return NSLKDD_ATTACK_GROUPS[label]
    raise ValueError(f'Unknown NSL-KDD label: {label}')


def _ignored_columns(row, label_col):
    ignored = {label_col}
    if label_col == len(row) - 2 and _is_float(row[-1]):
        ignored.add(len(row) - 1)
    return ignored


def _feature_rows(rows, label_col):
    feature_rows = []
    for row in rows:
        ignored = _ignored_columns(row, label_col)
        feature_rows.append([value for idx, value in enumerate(row) if idx not in ignored])
    return feature_rows


def _fit_feature_schema(feature_rows):
    num_cols = len(feature_rows[0])
    categorical_values = {}
    numeric_cols = []
    for col in range(num_cols):
        values = [row[col] for row in feature_rows]
        if all(_is_float(value) for value in values):
            numeric_cols.append(col)
        else:
            categorical_values[col] = sorted(set(values))
    return numeric_cols, categorical_values


def _transform_features(feature_rows, schema):
    numeric_cols, categorical_values = schema
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


def _build_features(rows, label_col, schema=None):
    feature_rows = _feature_rows(rows, label_col)
    if schema is None:
        schema = _fit_feature_schema(feature_rows)
    return _transform_features(feature_rows, schema), schema


def main():
    parser = argparse.ArgumentParser(description='Prepare NSL-KDD CSV/TXT data as 5-class EdgeQGFed npz.')
    parser.add_argument('--input', required=True, help='Path to raw NSL-KDD/flow CSV or TXT file.')
    parser.add_argument('--test', default=None, help='Optional raw NSL-KDD test CSV/TXT file.')
    parser.add_argument('--output', default='data/nslkdd/dataset.npz', help='Output npz path.')
    parser.add_argument('--users-column', type=int, default=-1, help='Optional user/host column index. Default: disabled.')
    args = parser.parse_args()

    rows = _read_rows(args.input)
    if not rows:
        raise ValueError(f'No rows found in {args.input}')

    label_col = _label_index(rows[0])
    test_rows = _read_rows(args.test) if args.test else []
    all_rows_for_schema = rows + test_rows
    all_label_col = _label_index(all_rows_for_schema[0])
    _, schema = _build_features(all_rows_for_schema, all_label_col)
    X, _ = _build_features(rows, label_col, schema=schema)
    y = np.asarray([_encode_label(row[label_col]) for row in rows], dtype=np.int64)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if test_rows:
        test_label_col = _label_index(test_rows[0])
        test_X, _ = _build_features(test_rows, test_label_col, schema=schema)
        test_y = np.asarray([_encode_label(row[test_label_col]) for row in test_rows], dtype=np.int64)
        np.savez_compressed(output_path, train_X=X, train_y=y, test_X=test_X, test_y=test_y)
    elif args.users_column >= 0:
        users = np.asarray([row[args.users_column] for row in rows])
        np.savez_compressed(output_path, X=X, y=y, users=users)
    else:
        np.savez_compressed(output_path, X=X, y=y)

    print(f'Saved {output_path} with X={X.shape}, y={y.shape}, classes={sorted(np.unique(y).tolist())}')


if __name__ == '__main__':
    main()
