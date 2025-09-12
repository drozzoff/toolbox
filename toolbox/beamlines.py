from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import json
import numpy as np
import scipy as sp
import pickle as pk
from toolbox import generate_bpsk, realign_mad_apertures


e0 = 931.49410242e6 # thus is an equivalent of the 1 amu in eV
a_e = 5.48579909065e-4


def _exc_freq_chirp(
	ctx,
	revolution_frequency: float,
	center_tune: float,
	tune_bandwidth: float,
	timestep: float, # or sampling rate
	excitation_duration: float | None,
	k0l: float,
	start_turn: int,
	**kwargs
	) -> xt.Exciter:

	excitation_frequency = revolution_frequency * center_tune
	width = revolution_frequency * tune_bandwidth

	time_stamps = np.arange(0, excitation_duration, timestep)

	samples = sp.signal.chirp(time_stamps, excitation_frequency - width, excitation_duration, excitation_frequency + width)

	return xt.Exciter(
		_context = ctx,
		samples = samples,
		sampling_frequency = 1 / timestep,
		duration = excitation_duration,
		frev = revolution_frequency,
		start_turn = start_turn,
		knl = [k0l],
		ksl = [0.0]
	)

def exc_freq_chirp(
	center_tune: float,
	tune_bandwidth: float,
	timestep: float, # or sampling rate
	excitation_duration: float | None,
	k0l: float,
	start_turn: int,
	**kwargs		
	) -> callable:
	
	def res(ctx, revolution_frequency: float):
		return _exc_freq_chirp(
			ctx = ctx,
			revolution_frequency = revolution_frequency,
			center_tune = center_tune,
			tune_bandwidth = tune_bandwidth,
			timestep = timestep, # or sampling rate
			excitation_duration = excitation_duration,
			k0l = k0l,
			start_turn = start_turn,
			**kwargs
		)
	
	return res

def _exc_bpsk(
	ctx,
	revolution_frequency: float,
	center_tune: float,
	tune_bandwidth: float,
	timestep: float, # or sampling rate
	excitation_duration: float | None,
	k0l: float,
	start_turn: int,
	**kwargs
	) -> xt.Exciter:

	excitation_frequency = revolution_frequency * center_tune
	chip_rate = revolution_frequency * tune_bandwidth

	timestamps, signal = generate_bpsk(
		f0 = excitation_frequency,
		timestep = timestep,
		signal_duration = excitation_duration,
		chip_rate = chip_rate,
		**kwargs
	)

	return xt.Exciter(
		_context = ctx,
		samples = signal,
		sampling_frequency = 1 / timestep,
		duration = excitation_duration,
		frev = revolution_frequency,
		start_turn = start_turn,
		knl = [k0l],
		ksl = [0.0]
	)

def exc_bpsk(
	center_tune: float,
	tune_bandwidth: float,
	timestep: float, # or sampling rate
	excitation_duration: float | None,
	k0l: float,
	start_turn: int,
	**kwargs		
	) -> callable:
	
	def res(ctx, revolution_frequency: float):
		return _exc_bpsk(
			ctx = ctx,
			revolution_frequency = revolution_frequency,
			center_tune = center_tune,
			tune_bandwidth = tune_bandwidth,
			timestep = timestep, # or sampling rate
			excitation_duration = excitation_duration,
			k0l = k0l,
			start_turn = start_turn,
			**kwargs
		)
	
	return res

# refeference_particles = {'mass':, 'charge', 'energy' (eV), 'kinetic_energy' (eV)}
def set_sis18ring(
	reference_particle: dict,
	qx: float | None =  None,
	qy: float | None = None,
	k2l_bends: float | None = None,
	exciter_setup = None, 
	**kwargs) -> xt.Line:
	"""
	Returns an `xt.Line` object containing SIS18 ring set up for the extraction.
	By default goes to a working point with the tune distance to the 3rd integer
	resonance of **-0.0037**.

	The following knobs are avaliable in the lattice:
	- `ks.amp`, Amplitude of the resonance exitation with sextupoles. Default = 0.04.
	- `ks.offset`, Strength of the chromaticity correction. Default = 0.0.
	- `ks.phase`, Phase of the sextupoles. Default = 0.35.
	- `es_bump.amp`. Amplitude of the ES septum orbit bump.
	- `ms_bump.amp`. Amplitude of the MS septum orbit bump.

	If `exciter_setup` is provided sets an `xt.Exciter` element with the provided
	signal (*chirp* or *bpsk* currently).
	"""
	
	carbon_beam_ref = xt.Particles(
		mass0 = reference_particle['mass'], 
		q0 = reference_particle['charge'], 
		kinetic_energy0 = reference_particle['kinetic_energy']
	)

	mad = Madx()
	mad.option(echo = False)

	mad.call("files/lattice/SIS18RING_xtrack.seq")
	mad.call("files/lattice/SIS18_TAU_1.000.str")

	mad.sequence.sis18ring.beam = {
		'particle': "ion",
		'mass': reference_particle['mass'] * 1e-9,
		'charge': reference_particle['charge'],
		'energy': reference_particle['energy'] * 1e-9    
	}

	mad.use(sequence = 'sis18ring')

	mad.call("files/lattice/SIS18_cryocatchers_new.str") # apertures

	sis18ring = xt.Line.from_madx_sequence(
		mad.sequence.sis18ring, 
		deferred_expressions = True,
		install_apertures = True,
		skip_markers = False,
		enable_align_errors = True
	)

	if kwargs.get('realign_apertures', True):
		realign_mad_apertures(sis18ring)

	sis18ring.particle_ref = carbon_beam_ref
	sis18ring.cycle(name_first_element = 'extraction_slow_begin')

	# Dipoles sextupolar components
	if k2l_bends is not None:
		bends, __ = sis18ring.get_elements_of_type(xt.Bend)
		for bend in bends:
			# it is devided by 2 because all the bends are split into 2
			# to accomodate the elements inside.
			bend.knl[2] = k2l_bends / 2

	# default setting
#	with open("files/settings/3rd_int_resonance_dist_0037.json", 'r') as f:
#		quads_settings = json.load(f)
#		for key in quads_settings:
#			sis18ring[key] = quads_settings[key]

	if qx is not None or qy is not None:
		# setting a working point
		tw = sis18ring.twiss4d()
		qx_target = qx if qx is not None else tw.qx
		qy_target = qy if qy is not None else tw.qy

		quads_vars = ["k1nl_gs01qs1f", "k1nl_gs01qs2d", "k1nl_gs12qs1f", "k1nl_gs12qs2d", "k1nl_gs12qs3t"]
    
		opt = sis18ring.match(
			method = '4d', 
			vary = [xt.VaryList(quads_vars, step = 1e-6, tag = 'quad')], 
			targets = [xt.TargetSet(qx = qx_target, qy = qy_target, tol = 0.1e-5, tag = 'qx_match')]
		)


	# Exciter setup
	if exciter_setup is not None:
		tw = sis18ring.twiss4d()

		revolution_frequency = 1 / tw.T_rev0

		sis18ring.discard_tracker()

		ctx = xo.ContextCpu(omp_num_threads = 'auto')
		rfko = exciter_setup(ctx = ctx, revolution_frequency = revolution_frequency)

		sis18ring.insert_element(
			index = 'gs01bo1eh',
			element = rfko, 
			name = 'rfko'
		)
		sis18ring.build_tracker(_context = ctx)

	# Knobs setup
	sis18ring.vars['ks.amp'] = 0.04 # amplitude of sext strengths | corresponds to samp in elegant
	sis18ring.vars['ks.phase'] = 0.3490658503988659 # phase in rad | corresponds to sphir in Elegant
	sis18ring.vars['ks.offset'] = 0.0 # offset | corresponds to choff in Elegant

	# sextupoles' integrated strengths (knobs)
	sextupoles_knobs = ["k2nl_gs01ks1c", "k2nl_gs03ks1c", "k2nl_gs05ks1c", "k2nl_gs07ks1c", "k2nl_gs09ks1c", "k2nl_gs11ks1c"]

	for i, knob in enumerate(sextupoles_knobs):
		sis18ring.vars[knob] = f'ks.amp * sin(2 * pi * {i} / 6 + ks.phase) + ks.offset'

	# ES septum fix, Section 6
	sis18ring['gs04me1e_ex_aper'].min_x = -0.055 - 7.4e-3 * 1.5

	sis18ring.vars['es_bump.amp'] = 0.0
	sis18ring.vars['ms_bump.amp'] = 0.0

	es_bump_knobs = {
		"k0nl_gs04mu1a": 0.115388600858,
		"k0nl_gs05mu1a": -0.011125411914, 
		"k0nl_gs05mu2a": 0.113037170557
	}

	for key in es_bump_knobs:
		sis18ring.vars[key] = f'es_bump.amp * {es_bump_knobs[key]}'


	ms_bump_knobs = {
		"k0nl_gs06mu1a": 0.185403373444,
		"k0nl_gs07mu1a": -0.0178872717494, 
		"k0nl_gs07mu2a": 0.181632238628
	}

	for key in ms_bump_knobs:
		sis18ring[key] = f'ms_bump.amp * {ms_bump_knobs[key]}'

	return sis18ring
