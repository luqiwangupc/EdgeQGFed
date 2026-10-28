"""Read-only audit tests: no benchmark, model or tracker access."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from omegaconf import OmegaConf

from tools.audit_e2_inputs import inspect_harbox_zip, inspect_run, main, partition_headers


class E2AuditTests(unittest.TestCase):
    def test_existing_files_do_not_prove_diagnostic_independence(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            config = OmegaConf.create({'datasets': {'name': 'cifar10', 'seed': 0, 'val_ratio': .1},
                                      'experiment': {'variant': 'sample_average'},
                                      'topology': {'num_clients': 2}})
            OmegaConf.save(config, run/'config_resolved.yaml')
            # Invalid model content must be safe: it is never deserialized.
            (run/'final.pth').write_bytes(b'not a model; never execute or deserialize')
            (run/'inputs.json').write_text(json.dumps({'client_edges': {'Client0': 'Edge0'}}))
            (run/'diagnostic_manifest.json').write_text('{}')
            (run/'metrics_by_checkpoint.jsonl').write_text(
                json.dumps({'step': 3000, 'split': 'fixed_horizon_test'})+'\n')
            result = inspect_run(run/'config_resolved.yaml')
            self.assertTrue(result['final_checkpoint_metadata']['exists'])
            self.assertFalse(result['checkpoint_deserialized'])
            self.assertTrue(result['fixed_horizon_test_record_found'])
            self.assertEqual(result['e2_independence_status'], 'UNCONFIRMED')
            self.assertIn('diagnostic_manifest.json', result['diagnostic_named_artifacts'])

    def test_numeric_headers_and_object_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'partition_masks.npz'
            np.savez(path, Client0_indices=np.arange(5), Client0_labeled_mask=np.ones(5, dtype=bool))
            report = partition_headers(path)
            self.assertEqual(report['index_rows_from_headers'], 5)
            self.assertFalse(report['content_read'])
            np.savez(path, Client0_indices=np.asarray([{'untrusted': 1}], dtype=object))
            with self.assertRaises(ValueError):
                partition_headers(path)

    def test_zip_inspection_never_reads_sensor_contents(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'harbox.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for user in range(120):
                    archive.writestr(f'HARBox/{user}/Walk_train.txt', 'not numeric sensor data')
                    archive.writestr(f'HARBox/{user}/Walk_test.txt', 'not numeric sensor data')
            with patch.dict('os.environ', {'HARBOX_TRAIN_USERS': '100'}):
                report = inspect_harbox_zip(path)
            self.assertEqual(report['by_filename_split']['train']['users_from_filenames_only'], 120)
            self.assertEqual(report['by_filename_split']['train']['hypothetical_train_prefix_count'], 100)
            self.assertFalse(report['sensor_records_read'])
            self.assertEqual(report['e2_independence_status'], 'UNCONFIRMED')
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(len(archive.namelist()), 240)

    def test_missing_roots_report_without_creating_files(self):
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root)/'missing'
            output = io.StringIO()
            with patch('sys.argv', ['audit', '--runs-root', str(missing)]), contextlib.redirect_stdout(output):
                status = main()
            self.assertEqual(status, 0)
            self.assertFalse(missing.exists())
            result = json.loads(output.getvalue())
            self.assertEqual(result['missing_roots'], [str(missing)])


if __name__ == '__main__':
    unittest.main()
