"""Pure-Python identity checks never invent a missing whole-file hash."""
import copy
import unittest

from scripts.q38_isoflops_contract import (
    validate_config_data_paths,validate_data_identity,runtime_identity_matches,validate_fit_grid,BUDGETS,GRID,
)
from tests.q38_isoflops_fixtures import data_identity,runtime_data_identity


class Contract(unittest.TestCase):
    def test_full_snapshot_and_metadata_without_meta_json_match(self):
        snapshot=data_identity()
        self.assertEqual(validate_data_identity(snapshot),snapshot)
        self.assertTrue(runtime_identity_matches(runtime_data_identity(snapshot),snapshot))

    def test_missing_or_changed_attestations_are_rejected(self):
        for stream,key,value in [('train','sha256','0'*64),('validation','sha256',None),
                                 ('train','mtime_ns',None),('validation','path','relative/file'),('train','bytes',4)]:
            changed=data_identity();changed[stream][key]=value
            with self.subTest(stream=stream,key=key),self.assertRaises(ValueError):
                validate_data_identity(changed)
        with self.assertRaises(ValueError):validate_data_identity(None)
        changed=data_identity();del changed['tokenizer_sha256']
        with self.assertRaises(ValueError):validate_data_identity(changed)

    def test_snapshot_does_not_default_missing_runtime_fields(self):
        snapshot=data_identity();actual=runtime_data_identity(snapshot)
        for stream in ('train','validation'):
            for key in ('path','bytes','mtime_ns'):
                changed=copy.deepcopy(actual);del changed[stream][key]
                self.assertFalse(runtime_identity_matches(changed,snapshot))

    def test_only_complete_predeclared_grid_is_accepted(self):
        points=[(BUDGETS[label],h) for label,widths in GRID.items() for h in widths]
        validate_fit_grid(points)
        for changed in [points[:-1],points+[points[0]],points[:-1]+[(3*10**17,512)]]:
            with self.assertRaises(ValueError):validate_fit_grid(changed)

    def test_config_streams_must_use_the_attested_canonical_paths(self):
        snapshot=data_identity()
        cfg={'train':{'data':snapshot['train']['path'],'val_data':snapshot['validation']['path']}}
        validate_config_data_paths(cfg,snapshot)
        for value in ['/data/q38/different-unhashed-pool/train.bin','/data/q38/alias/train.bin',
                      '/data/q38/fixture/../fixture/train.bin','/data//q38/fixture/train.bin',
                      'relative/train.bin']:
            changed=copy.deepcopy(cfg);changed['train']['data']=value
            with self.subTest(path=value),self.assertRaisesRegex(ValueError,'snapshot canonical path'):
                validate_config_data_paths(changed,snapshot)


if __name__=='__main__':unittest.main()
