import numpy as np
from rapidfuzz import fuzz

from src.sib2_context import first_number, siblings, split_features, street_text


def test_siblings_are_the_groups_best_other_rows():
    codes = np.array([0, 0, 0, 0, 0, 1, 1, 2])
    scores = np.array([0.1, 0.9, 0.5, np.nan, 0.7, 0.2, 0.3, 0.4], np.float32)
    sib = siblings(codes, scores, top=3)
    assert sib[0].tolist() == [1, 4, 2] and sib[1].tolist() == [4, 2, 0]      # best others by score, self skipped
    assert sib[5].tolist() == [6, -1, -1] and sib[7].tolist() == [-1, -1, -1]


def test_street_text_expands_abbreviations_and_drops_numbers():
    assert street_text("17 R. de Bruges") == street_text("17 Rue De Bruges") == "rue de bruges"
    assert street_text("N°16 Bd Saint-Michel") == "boulevard saint michel" and first_number("#0031 Main") == 31


def test_features_against_direct_computation():
    t_name = np.asarray(["Bordeaux Amicale SAS", "Bordeaux Amicale  SAS", "Bordeaux Amicale SAS", "Ectozeta"], dtype=object)
    t_addr = np.asarray(["17 Rue de Bruges", "17 R. DE BRUGES", "17 Rue Xaintrailles", "35 R du Brulis"], dtype=object)
    s_name = np.asarray(["Bordeaux Amicale SAS"] * 4, dtype=object)
    s_addr = np.asarray(["17 Rue de Bruges, Bordeaux"] * 4, dtype=object)
    sib = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1], [0, 1, 2]])
    X = split_features(t_name, t_addr, s_name, s_addr, sib)
    assert X[0, 1] == 1 and X[1, 1] == 0 and np.isclose(X[1, 0], fuzz.ratio(t_name[1], s_name[1]))
    assert X[2, 6] < X[1, 6]                                             # another street: weaker street consensus
    assert np.isclose(X[3, 3], 0) and X[0, 3] == 1                       # identical raw name among the siblings
    assert X[2, 8] == 1 and np.isclose(X[3, 8], 0)                       # same first number 17 / 35 vs 17
