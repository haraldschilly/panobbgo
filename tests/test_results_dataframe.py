
import unittest
import numpy as np
import pandas as pd
from panobbgo.core import Results
from panobbgo.lib import Result, Point, Problem
from panobbgo.utils import PanobbgoTestCase

class TestResultsDataFrame(PanobbgoTestCase):

    def setUp(self):
        super().setUp()
        # Use real Results object attached to the mock strategy
        self.strategy.results = Results(self.strategy)

    def test_dataframe_columns_no_collision(self):
        """
        Verify that scalar 'cv' and vector 'cv_vec' (renamed) do not collide
        in the Results DataFrame.
        """
        # Create a mock result with constraint vector
        r = Result(Point(np.array([0.0]), "test"), 0.0, cv_vec=np.array([1.0, 2.0]))

        self.strategy.results.add_results([r])
        df = self.strategy.results.results

        # Check column names
        # We expect ('cv_vec', 0), ('cv_vec', 1) and ('cv', 0)

        # 1. Check uniqueness of 'cv' access
        try:
            cv_col = df['cv']
            # Should be a Series or DataFrame with single column if accessed by top level
            # In pandas, if 'cv' is a top level key with only one sub-column '0',
            # df['cv'] returns a DataFrame with that sub-column.
            # If we had collision, it would return multiple columns or be ambiguous.

            # Since we renamed the vector to 'cv_vec', 'cv' should be unique (scalar).
            self.assertEqual(cv_col.shape[1], 1, "df['cv'] should return exactly one column (the scalar cv)")

            # Accessing scalar specifically
            scalar_val = float(df[('cv', 0)].iloc[0])
            self.assertAlmostEqual(scalar_val, r.cv)

        except KeyError:
            self.fail("Could not access 'cv' column")

        # 2. Check access to vector
        try:
            cv_vec_col = df['cv_vec']
            self.assertEqual(cv_vec_col.shape[1], 2, "df['cv_vec'] should return 2 columns")

            val0 = float(df[('cv_vec', 0)].iloc[0])
            val1 = float(df[('cv_vec', 1)].iloc[0])

            self.assertEqual(val0, 1.0)
            self.assertEqual(val1, 2.0)

        except KeyError:
            self.fail("Could not access 'cv_vec' column")

    def test_dataframe_unconstrained(self):
        """
        Verify behavior when no constraints are present.
        """
        r = Result(Point(np.array([0.0]), "test"), 0.0, cv_vec=None)

        self.strategy.results.add_results([r])
        df = self.strategy.results.results

        # Should have 'cv' (scalar 0) but no 'cv_vec'
        self.assertIn('cv', df.columns.levels[0])
        self.assertNotIn('cv_vec', df.columns.levels[0])

        scalar_val = float(df[('cv', 0)].iloc[0])
        self.assertEqual(scalar_val, 0.0)

    def test_get_history(self):
        """
        Verify that get_history returns correct numpy arrays.
        """
        # Create results
        r1 = Result(Point(np.array([1.0]), "t1"), 10.0, cv_vec=np.array([0.1]))
        r2 = Result(Point(np.array([2.0]), "t2"), 20.0, cv_vec=np.array([0.2]))

        self.strategy.results.add_results([r1, r2])

        # 1. Test full history
        h = self.strategy.results.get_history()

        self.assertEqual(len(h['x']), 2)
        self.assertTrue(np.allclose(h['x'], np.array([[1.0], [2.0]])))
        self.assertTrue(np.allclose(h['fx'], np.array([10.0, 20.0])))
        self.assertTrue(np.allclose(h['cv'], np.array([0.1, 0.2])))
        self.assertTrue(np.allclose(h['cv_vec'], np.array([[0.1], [0.2]])))
        self.assertTrue(np.array_equal(h['who'], np.array(['t1', 't2'])))

        # 2. Test limited history (n=1)
        h_last = self.strategy.results.get_history(n=1)
        self.assertEqual(len(h_last['x']), 1)
        self.assertTrue(np.allclose(h_last['x'], np.array([[2.0]])))
        self.assertTrue(np.allclose(h_last['fx'], np.array([20.0])))

    def test_get_history_empty(self):
        """
        Verify get_history on empty results.
        """
        h = self.strategy.results.get_history()
        self.assertEqual(len(h['x']), 0)
        self.assertEqual(len(h['fx']), 0)
        self.assertEqual(len(h['cv']), 0)
        self.assertEqual(len(h['cv_vec']), 0)

    def test_get_history_no_constraints(self):
        """
        Verify get_history handles unconstrained results correctly.
        """
        r = Result(Point(np.array([1.0]), "t1"), 10.0, cv_vec=None)
        self.strategy.results.add_results([r])

        h = self.strategy.results.get_history()
        self.assertEqual(len(h['cv_vec']), 1)
        self.assertEqual(h['cv_vec'].shape[1], 0) # Should be (1, 0)
        self.assertEqual(h['cv'][0], 0.0)

if __name__ == '__main__':
    unittest.main()
