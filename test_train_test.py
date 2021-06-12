from unittest import TestCase

from train_test import basic


class Testbasic(TestCase):
    def test_input_check(self):
        result = basic.input_check(5, 10)
        self.assertEqual(result, 5)

    def test_input_csv_2(self):
        result = basic.input_check(5, 10)
        self.assertEqual(result, 5)

    def test_input_csv(self):
        self.assertEqual(basic.input_check(5), "Image CSV folder")
