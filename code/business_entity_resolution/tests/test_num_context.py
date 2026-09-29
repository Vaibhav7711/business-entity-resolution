import numpy as np

from src.num_context import number_codes, pad, row_features, token_codes, MAX_NUMS, MAX_TOKS


def brute(s_addr, s_name, t_addr, t_name):
    sn, tn = number_codes(s_addr), number_codes(t_addr)
    st, tt = token_codes(s_name), token_codes(t_name)
    both = bool(sn) and bool(tn)
    shared = len(set(sn) & set(tn))
    union = len(set(sn) | set(tn))
    tshared, tunion = len(set(st) & set(tt)), len(set(st) | set(tt))
    return [float(tn[0] == sn[0]) if both else np.nan, shared if both else np.nan,
            shared / union if union else np.nan, float(len(tn) - shared) if tn else np.nan,
            tshared / tunion if tunion else np.nan, float(tt[0] == st[0]) if tt and st else np.nan]


def test_number_and_token_features_match_brute_force():
    cases = [("11311 127th Avenue, Kirkland, WA", "Pediatric Dental Physicians of Kirkland", "#10125 127th Avenue, Kirkland", "Dental Physicians Of Kirkland"),
             ("3100 Frontage Road", "Verray Sizzle Inc", "3100-3104 FRONTAGE RD", "VERRAY SIZZLE INC"),
             ("44 Gorski Street", "Atlantic Microelectronics Inc", "", "Atlantic Microelectronics"),
             ("", "Lyrium Roman LLC", "", "Lyrium Raoyal LLC"),
             ("H.No 031, Anurag Nagar", "Corporate Services Private Limited", "H.no 31 Anurag", "Corporate Services"),
             ("1 2 3 4 5 6", "A B", "6 5 4", "B A")]
    s_nums = pad([number_codes(c[0]) for c in cases], MAX_NUMS); s_toks = pad([token_codes(c[1]) for c in cases], MAX_TOKS)
    t_nums = pad([number_codes(c[2]) for c in cases], MAX_NUMS); t_toks = pad([token_codes(c[3]) for c in cases], MAX_TOKS)
    got = row_features(s_nums, s_toks, t_nums, t_toks)
    want = np.asarray([brute(*c) for c in cases], np.float32)
    assert np.allclose(got, want, equal_nan=True)
    assert got[0, 0] == 0 and got[1, 0] == 1 and got[4, 0] == 1        # 031 == 31; 10125 != 11311
    assert np.isnan(got[2, 0]) and np.isnan(got[3, 2])                 # no number on one / both sides
