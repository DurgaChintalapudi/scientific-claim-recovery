"""Regression tests: interpretation of author annotation labels.

These are synthetic fixtures, not empirical scientific evidence.
"""
import csv
import pytest
from recovery.data import Case
from recovery.evaluate import read_annotations, score


def case():
    return Case(case_id='synthetic', claim_id=1, doc_id=1, claim='A claim',
                title='Synthetic', evidence=['A sentence'], rationale_indices=[0],
                gold_label='SUPPORT', source_split='dev')


def annotated(label, support='SUPPORTED'):
    out={'case_id':'synthetic', 'policy':'B2', 'action':'REPAIR',
         'final_claim':'Rewritten claim', 'model_calls':2}
    ann={'case_id':'synthetic','policy':'B2', 'final_support':support,
         'meaning_preservation':label}
    return score([out],[case()],[ann])['B2']


def test_preserved_counts_as_useful():
    assert annotated('PRESERVED')['human_useful_coverage'] == 1.0


def test_partial_and_lost_are_excluded():
    assert annotated('PARTIAL')['human_useful_coverage'] == 0.0
    assert annotated('LOST')['human_useful_coverage'] == 0.0


def test_unsupported_preserved_is_not_useful():
    assert annotated('PRESERVED',support='CONTRADICTED')['human_useful_coverage']==0.0


def test_legacy_label_supported():
    assert annotated('MEANINGFUL')['human_useful_coverage']==1.0


def test_reject_duplicate_annotations(tmp_path):
    path=tmp_path/'annotations.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','policy','final_support','meaning_preservation'])
        w.writeheader()
        for _ in range(2):w.writerow({'case_id':'synthetic','policy':'B2','final_support':'SUPPORTED','meaning_preservation':'PRESERVED'})
    with pytest.raises(ValueError,match='Duplicate annotation'):read_annotations(path)
