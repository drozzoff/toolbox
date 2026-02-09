import socket
import time
import json
import numpy as np
import random


class DummyGenerator:

	def __init__(self, host: str, port: int):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.connect((host, port))

	def stream(self, buffers_to_stream: list[str], time_delta: float = 0.5):
		_current_turn, step = 0, 100
		while True:
			data = {}
			for key in buffers_to_stream:
				# turn and turn-by-turn data
				if key in ['turn']:
					data[key] = np.arange(_current_turn, _current_turn + step).tolist()
				if key in ['Nparticles', 'lost_on_septum_wires']:
					data[key] = np.random.normal(size = step).tolist()

				if key in ['extracted_at_ES:at_turn']:
					N_elems = None
					for key_ in {'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'}:
						N_elems = data.get(key_, N_elems)
					if N_elems is None:
						N_elems = random.randint(1, 100)
					
					data[key] = (_current_turn + np.random.randint(step, size = N_elems)).tolist()

				if key in ['extracted_at_ES:x', 'extracted_at_ES:px']:
					N_elems = None
					for key_ in {'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'}:
						N_elems = data.get(key_, N_elems)
					if N_elems is None:
						N_elems = random.randint(1, 100)

					data[key] = np.random.normal(size = step).tolist()
				
				#print(data)
			self.sock.sendall((json.dumps(data) + '\n').encode())
			_current_turn += step
			time.sleep(time_delta)
