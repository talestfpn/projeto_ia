import unittest

import numpy as np

import main


class FakeEncoder:
    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray([probabilities], dtype=float)

    def predict_proba(self, values):
        return self.probabilities


class DualHeadContractTest(unittest.TestCase):
    def setUp(self):
        self.previous_state = dict(main.STATE)
        main.STATE.clear()
        main.STATE.update(
            encoder=FakeEncoder(),
            clf=None,
            sep="[SEP]",
            classes=np.array(["cs.AI", "cs.CL", "cs.LG"], dtype=object),
            X=np.array([[1.0, 0.0]], dtype=np.float32),
            meta=[{"id": "1234.5678", "title": "Paper", "category": "cs.LG"}],
            inference_mode="calibrated_multilabel_40000",
            bundle={
                "primary_model": FakeModel([0.20, 0.25, 0.55]),
                "multilabel_model": FakeModel([0.70, 0.45, 0.80]),
                "primary_classes": ["cs.AI", "cs.CL", "cs.LG"],
                "multilabel_classes": ["cs.AI", "cs.CL", "cs.LG"],
                "thresholds": {"cs.AI": 0.60, "cs.CL": 0.50, "cs.LG": 0.85},
            },
        )

    def tearDown(self):
        main.STATE.clear()
        main.STATE.update(self.previous_state)

    def test_primary_ranking_and_independent_multilabel_thresholds(self):
        result = main.classify(main.ClassifyIn(abstract="Scientific abstract", top_k=3, similar_k=1))

        self.assertEqual(result.predicted_category, "cs.LG")
        self.assertAlmostEqual(result.confidence, 0.55)
        self.assertEqual([item.category for item in result.ranking], ["cs.LG", "cs.CL", "cs.AI"])

        applicable = {item.category: item.score for item in result.applicable_categories}
        self.assertEqual(set(applicable), {"cs.LG", "cs.AI"})
        self.assertAlmostEqual(applicable["cs.LG"], 0.80)
        self.assertAlmostEqual(applicable["cs.AI"], 0.70)
        self.assertNotIn("cs.CL", applicable)


if __name__ == "__main__":
    unittest.main()
