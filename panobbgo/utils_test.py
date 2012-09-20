import unittest
import numpy as np
from panobbgo.utils import is_right, is_left

class TestUtils(unittest.TestCase):

  def setUp(self):
    pass

  def test_is_left(self):
    # simplified
    p0 = np.array([5,5])
    p1 = np.array([7,21])
    pt = np.array([5, 5.01])
    self.assertTrue(is_left(p0, p1, pt))
    pt = p1
    self.assertFalse(is_left(p0, p1, pt))

  def test_is_right_vertical(self):
    p0 = np.array([ 1, 1])
    p1 = np.array([ 1, 3])
    testpoints = [
        np.array([1.1,  2.2]),
        np.array([1.1, -2.2]),
        np.array([5  ,  2  ])
      ]

    for tp in testpoints:
      self.assertTrue(is_right(p0, p1, tp), "%s" % tp)
      tp = p1
      self.assertFalse(is_left(p0, p1, tp))

  def test_is_right_diagonal(self):
    p0 = np.array([ 0, 2])
    p1 = np.array([ 1, 3])
    testpoints = [
        np.array([  .5, 1]),
        np.array([ 1.2, 3.1])
      ]

    for tp in testpoints:
      self.assertTrue(is_right(p0, p1, tp), "%s" % tp)

  def test_is_right_horizontal(self):
    p0 = np.array([ 2, 2])
    p1 = np.array([ 3, 2])
    testpoints = [
        np.array([  2, 2.2]),
        np.array([  0,   5])
      ]

    for tp in testpoints:
      self.assertFalse(is_right(p0, p1, tp), "%s" % tp)

  def test_shuffle(self):
    #self.assertEqual(self.seq, range(10))
    # should raise an exception for an immutable sequence
    #self.assertRaises(TypeError, random.shuffle, (1,2,3))
    #self.assertTrue(element in self.seq)
    pass

if __name__ == '__main__':
    unittest.main()
