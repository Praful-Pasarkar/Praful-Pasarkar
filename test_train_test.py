from unittest import TestCase

import pytest

from train_test import basic


class Testbasic(TestCase):
    def test_input_check(self):
        result = basic.input_check(5, 10)
        self.assertEqual(result, 5, 'They are equal')

    def test_input_csv_2(self):
        result = basic.input_csv_2(5, 10)
        self.assertEqual(result, 5, 'They are equal')

    def test_input_csv(self):
        self.assertEqual(basic.input_csv(5), "Image CSV folder", 'They are equal')

    def test_check(self):
        result = basic.check(5)
        b = 5
        self.assertEqual(result, b, 'They are equal')