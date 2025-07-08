import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_dilation
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tqdm

bin_width = 0.000244140625

def estimate_noise(data: pd.Series, noise_percentile: float = 0.9) -> tuple[float, float]:
	cutoff = np.quantile(data, noise_percentile)

	pure_noise = data[data <= cutoff]

	floor = np.median(pure_noise)
	sigma = pure_noise.std(ddof = 1)

	return floor, sigma

def estimate_search_region(data: pd.Series, sigma_range: float = 3, noise_percentile: float = 0.9) -> list[float]:

	collapsed_no_dc = data.sum(axis = 0).drop(0.0)

	mu0 = collapsed_no_dc.idxmax()

	global_floor, __ = estimate_noise(collapsed_no_dc, noise_percentile)

	baseline = collapsed_no_dc - global_floor
	top_part = baseline[baseline > baseline.max() / 2]

	fwhm = top_part.index[-1] - top_part.index[0]
	sigma0 = fwhm / 2.355

	search_region = [mu0 - sigma_range * sigma0, mu0 + sigma_range * sigma0]

	return search_region

def get_peak_candidates(
		data: pd.Series, 
		search_region: list[float],
		noise_percentile: float = 0.95, 
		noise_width: float = 3, # in sigmas
		peak_min_length: int = 3, # in bins
		post_dilation: int | None = None
	) -> pd.Series | None:

	floor, noise_sigma = estimate_noise(data, noise_percentile)
	signal_mask = data > (floor + noise_width * noise_sigma)

	# search window given by `search_region`
	idx = pd.Series(data.index, index = data.index)
	search_window_mask = (idx >= search_region[0]) & (idx <= search_region[1])

	signal_in_window_mask = signal_mask & search_window_mask

	closed_signal_in_window_mask = pd.Series(
		binary_closing(signal_in_window_mask, structure = np.ones(3, bool)), 
		index = data.index
	)

	bin_ids = [int(x) for x in data[closed_signal_in_window_mask].index / bin_width]
	gaps = np.diff(bin_ids)

	split_locs = np.where(gaps > 1)[0] + 1
	if len(split_locs) != 0:
		warnings.warn("Multiple peaks identified")
		return None
		
	if len(closed_signal_in_window_mask[closed_signal_in_window_mask == True]) < peak_min_length:
		warnings.warn(f"Length of the peak is smaller than peak_min_length = {peak_min_length}")
		return None

	# signal dilation
	if post_dilation is None:
		return closed_signal_in_window_mask
	elif isinstance(post_dilation, int):
		# post_dilation is considered a radius of the dilation
		dilated_closed_signal_in_window_mask = pd.Series(
			binary_dilation(closed_signal_in_window_mask, structure = np.ones(1 + 2 * post_dilation, bool)), 
			index = data.index
		)
		return dilated_closed_signal_in_window_mask
	else:
		pass
		return None


def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

rational_weight = lambda x, noise_sigma: x**2 / (x**2 + noise_sigma**2)

exponential_ramp = lambda x, noise_sigma, k: 1 - np.exp(-(x / noise_sigma)**k )
exponential_ramp2 = lambda x, noise_sigma, k: 1 - np.exp(-(x / (2 * noise_sigma))**k )

exp_ramp_1 = lambda x, noise_sigma: exponential_ramp(x, noise_sigma, 1)
exp_ramp_2 = lambda x, noise_sigma: exponential_ramp(x, noise_sigma, 2)
exp_ramp_3 = lambda x, noise_sigma: exponential_ramp(x, noise_sigma, 3)

exp_ramp2_2 = lambda x, noise_sigma: exponential_ramp2(x, noise_sigma, 2)

def residual(p, x, y, weight):
	A, mu, sigma = p

	y_fit = A * np.exp(-0.5*((x - mu)/sigma)**2)
	r = np.sqrt(weight)*(y - y_fit)

	#cost = 0.5 * np.dot(r, r)
	#_history.append(( _iteration, A, mu, sigma, cost))
	#_iteration += 1

	return r

def fit_peak_with_gaussian(
	data: pd.Series, 
	noise_sigma: float,
	weights_func: callable,
	weighted_score: bool = True,
	plot_to_debug: bool = False
	) -> dict:
	'''`data` is expected to be brought to the noise floor'''

	x_data = data.index.values
	y_data = data.values

	A0 = data.max()
	mu0 = data.index[data.argmax()]
	sigma0 = (data.index[-1] - data.index[0]) / 2

	lower = [0, data.index[0], 0]
	upper = [np.inf, data.index[-1], (data.index[-1] - data.index[0]) * 0.5]

	positive_y_data = np.clip(y_data, a_min = 0, a_max = None)
	w = np.array(list(map(lambda x: weights_func(x, noise_sigma), positive_y_data)))
		
	res = least_squares(
		fun = residual,
		x0 = [A0, mu0, sigma0],
		args = (x_data, y_data, w),
		method = 'trf',
		x_scale = np.array([A0, 1.0, sigma0]),
		bounds = (lower, upper),
	#        verbose = 2,
	)
		
	J = res.jac
	cost = res.cost

	N, m = J.shape

	dof = max(1, N - m)
	s2  = (2 * cost) / dof
	JTJ = J.T @ J
	cov = np.linalg.inv(JTJ) * s2

	sigma_mu = np.sqrt(cov[1, 1])

	A_fit, mu_fit, sig_fit = res.x

	y_pred = np.array(list(map(lambda x: gauss(x, *res.x), x_data)))
	score = None
	if weighted_score:
		score = r2_score(y_data, y_pred, sample_weight = w)
	else:
		score = r2_score(y_data, y_pred, sample_weight = w)

	fitted_gauss = lambda x: gauss(x, A_fit, mu_fit, sig_fit)

	x_fitted_gauss = np.linspace(x_data[0], x_data[-1], 100)
	y_fitted_gauss = list(map(lambda x: fitted_gauss(x), x_fitted_gauss))

	if plot_to_debug:
		plt.figure()

		data.plot(label = "data")

		plt.plot(x_fitted_gauss, y_fitted_gauss, color = "red", label = "fit")
		plt.legend()
		plt.show()

	return dict(score = score, center = mu_fit, error = sigma_mu)

def parse_tunes(
	data: pd.DataFrame,
	noise_percentile: float = 0.9,
	search_region_width: float = 5, # in sigmas
	noise_width: float = 3, # in sigmas
	peak_min_length: int = 3, # in bins
	post_dilation: int = 3, # radius of dilation in bins for the fit
	weights_func: callable = lambda x: x, # weights function for the fit
	r2_score_cut = 0.75, # better fits survive
	verbose: int = 0 # control the data printed
	) -> pd.DataFrame:
	"""
	Parse the spectrogram `data`. It finds the range of the tunes where the signal could be,
	evaluates the potential peaks for each timestamp and does gaussian fit. Returns the fits
	with the best scores.
	"""
	fit_summary = {
		'score': [],
		'center': [],
		'error': []
	}

	for timestamp in tqdm(data.index):
		row = data.loc[timestamp]
		note = None
		
		noise, noise_sigma = estimate_noise(row, noise_percentile)
		
		search_region = estimate_search_region(data, search_region_width)

		with warnings.catch_warnings(record = True) as w:
			warnings.simplefilter("always")
			peak_mask = get_peak_candidates(
				data = row, 
				search_region = search_region,
				noise_percentile = noise_percentile,
				noise_width = noise_width, # in sigmas
				peak_min_length = peak_min_length, # in bins
				post_dilation = post_dilation
			)
		if w:
			note = "; ".join(str(wrn.message) for wrn in w)

		if peak_mask is None:
			fit_summary['score'].append(None)
			fit_summary['center'].append(None)
			fit_summary['error'].append(None)
			if verbose > 1:
				print(f"{timestamp}, No peak:", note or "")
				
			continue
		
		fit_res = fit_peak_with_gaussian(
			data = row[peak_mask] - noise,
			noise_sigma = noise_sigma,
			weights_func = weights_func,
			plot_to_debug = False
		)
		fit_summary['score'].append(fit_res['score'])
		fit_summary['center'].append(fit_res['center'])
		fit_summary['error'].append(fit_res['error'])
		
		if verbose > 0:
			msg = f"{timestamp}, fit score = {fit_res['score']}, mu = {fit_res['center']} " \
					f"+/- {fit_res['error']}"
			if note:
				msg += f", warning: {note}"
			print(msg)

	summary = pd.DataFrame(fit_summary, index = data.index)

	res = summary[summary.score >= r2_score_cut]

	return res

if __name__ == "__main__":
	"""An example, parsing the spectrum"""

	ks_offset = "0.14"
	ks_amp = "0.06"
	dp = "-1e-3"


	p_0 = pd.read_pickle(f"tune_measurements/noise excitation/ks.offset = {ks_offset}/ks.amp = {ks_amp}/{dp} offset/horizontal_spectrum.pkl")
	path = f"tune_measurements/noise excitation/ks.offset = {ks_offset}/ks.amp = {ks_amp}/{dp} offset/"

	res = parse_tunes(
		data = p_0,
		noise_percentile = 0.99,
		search_region_width = 5,
		noise_width = 3,
		peak_min_length = 1,
		post_dilation = 3,
		weights_func = rational_weight,
		r2_score_cut = 0.5,
		verbose = 2
	)

	plt.figure()

	plt.errorbar(
		res.center,
		res.index,
		xerr = res.error,
		fmt = 'o',
		capsize = 3,
		color = "blue",
		label = "Reconstructed peaks"
	)

	mean = np.sum(res.center) / len(res.center)
	var = np.sum((res.center - mean)**2) / len(res.center)
	sigma = np.sqrt(var)
	print(f"Normal: {mean} +/- {sigma}")

	w = 1.0 / res.error**2
	w_mean = np.sum(res.center * w)/ np.sum(w)
	var = np.sum(w * (res.center - w_mean)**2) / np.sum(w)
	w_sigma = np.sqrt(var)
	print(f"Wighted: {w_mean} +/- {w_sigma}")

	plt.axvline(x = mean, color = "green", label = f"Avg., {round(mean, 5)} +/- {round(sigma, 5)}")
	plt.axvspan(mean - sigma, mean + sigma, color = 'green', alpha = 0.2)

	plt.axvline(x = w_mean, color = "red", label = f"W.avg., {round(w_mean, 5)} +/- {round(w_sigma, 5)}")
	plt.axvspan(w_mean - w_sigma, w_mean + w_sigma, color = 'red', alpha = 0.2)


	timestamp_bins = 5
	timestamp_step = (res.index.max() - res.index.min()) / timestamp_bins

	for i in range(timestamp_bins):
		mask = np.logical_and(res.index - res.index.min() >= timestamp_step * i, res.index - res.index.min() <= timestamp_step * (i + 1))

		tunes = res.center[mask]
		tunes_errors = res.error[mask]

		x_mean = np.sum(tunes) / len(tunes)
		
		var = np.sum((tunes - x_mean)**2) / len(tunes)
		sigma = np.sqrt(var)

	#    print(x_mean, sigma)
		plt.fill_betweenx(
			[res.index.min() + timestamp_step * i, res.index.min() + timestamp_step * (i + 1)], 
			x_mean - sigma, 
			x_mean + sigma, 
			color = "navy", 
			alpha = 0.3
		)

	plt.ylim(-0.01, 0.36)
	plt.legend()
	plt.show()