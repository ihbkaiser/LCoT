import unittest

import pandas as pd

from finite_cot.frontier_validation import (
    decision_report, donor_flip_summary, primary_summary, secondary_summary,
    theoretical_error_envelope,
)


class FrontierValidationTests(unittest.TestCase):
    def records(self):
        rows=[]
        vectors={"a":[0,1,0,1],"b":[1,0,1,0]}
        for seed in (17,42):
            for prefix,bits in vectors.items():
                for query,label in enumerate(bits):
                    rows.append({
                        "model":"m","seed":seed,"split":"audit","n":4,"d":4,"p":1,"L":0,"T":1,
                        "prefix_id":f"{seed}-{prefix}","query":query,"label":label,"prediction":label,
                        "prob_1":0.99 if label else 0.01,"state_key":f"state-{prefix}","transcript_key":"",
                        "bit_hash":prefix,"saturation_rate":1.0,
                    })
        return pd.DataFrame(rows)

    def test_primary_and_secondary(self):
        q=self.records()
        primary=primary_summary(q,bootstrap_draws=100)
        secondary=secondary_summary(q)
        self.assertEqual(primary.loc[0,"mean_bit_error"],0.0)
        self.assertEqual(primary.loc[0,"theoretical_envelope"],0.0)
        self.assertEqual(secondary.loc[0,"exact_recovery"],1.0)
        self.assertEqual(secondary.loc[0,"collision_conflict_rate"],0.0)

    def test_collision_conflict(self):
        q=self.records(); q["state_key"]="collision"
        secondary=secondary_summary(q)
        self.assertGreater(secondary.loc[0,"collision_conflict_rate"],0.0)

    def test_donor_flip(self):
        rows=pd.DataFrame([{
            "model":"m","seed":17,"n":4,"d":4,"p":1,"L":0,"T":1,"prefix_id":"a",
            "patch_channel":"state","recipient_prediction":0,"patched_prediction":1,
            "recipient_label":0,"donor_label":1,
        }])
        self.assertEqual(donor_flip_summary(rows).loc[0,"donor_consistent_flip_rate"],1.0)

    def test_envelope(self):
        self.assertAlmostEqual(theoretical_error_envelope(0.0),0.5)
        self.assertEqual(theoretical_error_envelope(1.0),0.0)

if __name__ == "__main__":
    unittest.main()
