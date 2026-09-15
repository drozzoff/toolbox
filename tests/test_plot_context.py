import json
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox.plotting import PlotContext


class PlotContextStateTests(unittest.TestCase):
	def setUp(self):
		self.style_file = tempfile.NamedTemporaryFile(mode = "w", suffix = ".json")
		json.dump({"Figure_size": {"width": 6, "height": 4}}, self.style_file)
		self.style_file.flush()

		self.context = PlotContext(
			style = self.style_file.name,
			show_survey = False,
			show_apertures = False,
		)

	def tearDown(self):
		plt.close(self.context.fig)
		self.style_file.close()

	def test_explicit_limits_are_reset_before_the_next_frame(self):
		initial_xlim = self.context.main_subplot.get_xlim()
		initial_ylim = self.context.main_subplot.get_ylim()

		with self.context:
			self.context.add_plot([0, 1], [0, 1])
			self.context.main_subplot.set_xlim(10, 20)
			self.context.main_subplot.set_ylim(30, 40)

		with self.context:
			self.assertEqual(self.context.main_subplot.get_xlim(), initial_xlim)
			self.assertEqual(self.context.main_subplot.get_ylim(), initial_ylim)

	def test_removed_data_does_not_affect_the_next_frame_autoscaling(self):
		with self.context:
			self.context.add_plot([0, 100], [0, 100])
			self.context.fig.canvas.draw()

		with self.context:
			self.context.add_plot([0, 1], [0, 1])
			self.context.fig.canvas.draw()

		self.assertAlmostEqual(self.context.main_subplot.get_xlim()[1], 1.05)
		self.assertAlmostEqual(self.context.main_subplot.get_ylim()[1], 1.05)

	def test_commit_restores_the_static_view(self):
		initial_xlim = self.context.main_subplot.get_xlim()

		with self.context:
			self.context.add_plot([0, 1], [0, 1])
			self.context.main_subplot.set_xlim(10, 20)

		self.context.commit()

		self.assertEqual(self.context.main_subplot.get_xlim(), initial_xlim)


if __name__ == "__main__":
	unittest.main()
