from rich.table import Table
from rich.live import Live
from typing import List

class tri_sep:

    def __init__(self, line, ex_norm: float):
        self.line = line
        self.x_norm_tri = None
        self.x_norm_sep = None

        self.ex_norm = ex_norm

    def get_particle(self, x_in: float):
        return self.line.build_particles(
            x_norm = x_in,
            px_norm =  0,
            y_norm = 0,
            py_norm = 0,
            nemitt_x = self.ex_norm,
            nemitt_y = 0.0,
            zeta = 0.0,
            delta = 0.0,
            mode = 'normalized_transverse',
            method = '4d'
        )
    
    
    def get_tri_sep_particle(self):
        return self.get_particle(self.x_norm_tri), self.get_particle(self.x_norm_sep)
            
    
    def search_separatrix(self, search_region: List, num_turns: int = 2000, precision: float = 1e-7, **kwargs):
        """
        Search for the separatrix. If the particle survives `num_turns` it is considered as stable.
        Otherwise - unstable. In the loop `search_region` is shrinked such that the first value
        corresponds to a particle on the triangle and second - on the separatrix.
        The function returns 2 `xt.Particles` objects, first corresponds to a particle on the 
        stable triangle, second - on the separatrix.
        
        Parameters
        ----------
        search_region
            Normalized coordinate in sigmas.
        num_turns
            Number of the turns for the tracking
        precision
            The desired width in the norm. phase space between the stable and unstable particle
    
        Additional parameters
        ---------------------
        
        
        Return
        ------
        xt.Particles, xt.Particles
            A pair of particles, on a stable triangle and on separatrix        
        """
        max_iter = kwargs.get('max_iter', 100)
    
        table = Table(title = "Separatrix search")
        table.add_column("x_test", style = "black")
        table.add_column("Stable?", style = "magenta")
        table.add_column("Precision", style = "green", justify = "right")
        
        def is_stable(turn):
            return turn == num_turns
    
        i = 0
        with Live(table, refresh_per_second = 10) as live:
            while (search_region[1] - search_region[0] > precision) and (i < max_iter):
                x_test = sum(search_region) / 2
                
                particle = self.get_particle(x_test)
                
                self.line.track(particle, num_turns = num_turns, turn_by_turn_monitor = True)
                rec = self.line.record_last_track
                
                table.add_row(str(format(x_test, ".7e")), str(is_stable(particle.at_turn[0])), str(format(search_region[1] - search_region[0], ".2e")))
                
                if is_stable(particle.at_turn[0]):
                    # x_test cooresponds to a stable particle
                    search_region[1] = x_test
                else:
                    # x_test corresponds to an unstable particle
                    search_region[0] = x_test     
    
                i += 1
                live.refresh()

        if search_region [0] < 0.0:
            self.x_norm_sep = search_region[0]
            self.x_norm_tri = search_region[1]
        else:
            self.x_norm_sep = search_region[1]
            self.x_norm_tri = search_region[0]