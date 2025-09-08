import numpy as np
from scipy.interpolate import interp1d
import os

class EndfParser:
    """
    Class for parsing and processing ENDF/B nuclear data.
    Provides methods for accessing cross sections, scattering laws, and fission spectra.
    """
    
    def __init__(self):
        """Initialize the ENDF parser."""
        self.data_dir = "endf_data"  # Directory containing ENDF data files
        self.isotopes = {}  # Dictionary to store isotope data
        self.energy_grid = np.logspace(-5, 7, 1000)  # Energy grid from 1e-5 eV to 1e7 eV
        
    def load_isotope(self, isotope):
        """
        Load isotope data from ENDF file.
        For now, using simplified data based on literature values.
        
        Parameters:
        -----------
        isotope : str
            Isotope identifier (e.g., 'U235', 'U238', 'H2O', 'D2O', 'C')
        """
        # Default cross-section values (based on literature)
        if isotope == 'U235':
            self.isotopes[isotope] = {
                'fission': self._create_fission_xs(),
                'capture': self._create_capture_xs(),
                'scatter': self._create_scatter_xs(),
                'scattering_law': self._create_scattering_law(),
                'fission_spectrum': self._create_fission_spectrum()
            }
        elif isotope == 'U238':
            self.isotopes[isotope] = {
                'fission': self._create_fission_xs(u238=True),
                'capture': self._create_capture_xs(u238=True),
                'scatter': self._create_scatter_xs(),
                'scattering_law': self._create_scattering_law(),
                'fission_spectrum': self._create_fission_spectrum(u238=True)
            }
        elif isotope in ['H2O', 'D2O', 'C']:
            self.isotopes[isotope] = {
                'fission': np.zeros_like(self.energy_grid),
                'capture': self._create_capture_xs(moderator=isotope),
                'scatter': self._create_scatter_xs(moderator=isotope),
                'scattering_law': self._create_scattering_law(moderator=isotope),
                'fission_spectrum': None
            }
        else:
            raise ValueError(f"Unsupported isotope: {isotope}")
    
    def get_cross_sections(self, isotope, energy):
        """
        Get cross sections for given isotope and energy.
        
        Parameters:
        -----------
        isotope : str
            Isotope identifier
        energy : float
            Neutron energy in eV
            
        Returns:
        --------
        dict
            Dictionary containing cross sections in barns
        """
        if isotope not in self.isotopes:
            self.load_isotope(isotope)
            
        # Interpolate cross sections
        xs = {}
        for reaction in ['fission', 'capture', 'scatter']:
            xs[reaction] = np.interp(energy, self.energy_grid, 
                                   self.isotopes[isotope][reaction])
        
        return xs
    
    def get_scattering_law(self, isotope, energy):
        """
        Get scattering law parameters for given isotope and energy.
        
        Parameters:
        -----------
        isotope : str
            Isotope identifier
        energy : float
            Neutron energy in eV
            
        Returns:
        --------
        float
            Average logarithmic energy decrement
        """
        if isotope not in self.isotopes:
            self.load_isotope(isotope)
            
        return np.interp(energy, self.energy_grid, 
                        self.isotopes[isotope]['scattering_law'])
    
    def get_fission_spectrum(self, isotope):
        """
        Get fission neutron energy spectrum.
        
        Parameters:
        -----------
        isotope : str
            Isotope identifier
            
        Returns:
        --------
        np.ndarray
            Fission spectrum probabilities
        """
        if isotope not in self.isotopes:
            self.load_isotope(isotope)
            
        return self.isotopes[isotope]['fission_spectrum']
    
    def _create_fission_xs(self, u238=False):
        """Create energy-dependent fission cross section."""
        if u238:
            # U-238 fission cross section
            xs = np.zeros_like(self.energy_grid)
            # Add small fast fission contribution
            fast_mask = self.energy_grid > 1e6
            xs[fast_mask] = 0.3 * np.exp(-self.energy_grid[fast_mask]/2e6)
        else:
            # U-235 fission cross section
            xs = np.zeros_like(self.energy_grid)
            # Thermal region
            thermal_mask = self.energy_grid <= 0.025
            xs[thermal_mask] = 582.2
            # Resonance region
            resonance_mask = (self.energy_grid > 0.025) & (self.energy_grid <= 1000)
            xs[resonance_mask] = 100.0 * np.exp(-self.energy_grid[resonance_mask]/100)
            # Fast region
            fast_mask = self.energy_grid > 1000
            xs[fast_mask] = 1.0
            
        return xs
    
    def _create_capture_xs(self, u238=False, moderator=None):
        """Create energy-dependent capture cross section."""
        if moderator:
            if moderator == 'H2O':
                xs = 0.664 * np.sqrt(0.025/self.energy_grid)
            elif moderator == 'D2O':
                xs = 0.001 * np.sqrt(0.025/self.energy_grid)
            elif moderator == 'C':
                xs = 0.0032 * np.sqrt(0.025/self.energy_grid)
        else:
            if u238:
                # U-238 capture cross section
                xs = np.zeros_like(self.energy_grid)
                # Thermal region
                thermal_mask = self.energy_grid <= 0.025
                xs[thermal_mask] = 2.72
                # Resonance region
                resonance_mask = (self.energy_grid > 0.025) & (self.energy_grid <= 1000)
                xs[resonance_mask] = 280.0 * np.exp(-self.energy_grid[resonance_mask]/100)
                # Fast region
                fast_mask = self.energy_grid > 1000
                xs[fast_mask] = 0.07
            else:
                # U-235 capture cross section
                xs = np.zeros_like(self.energy_grid)
                # Thermal region
                thermal_mask = self.energy_grid <= 0.025
                xs[thermal_mask] = 98.96
                # Resonance region
                resonance_mask = (self.energy_grid > 0.025) & (self.energy_grid <= 1000)
                xs[resonance_mask] = 140.0 * np.exp(-self.energy_grid[resonance_mask]/100)
                # Fast region
                fast_mask = self.energy_grid > 1000
                xs[fast_mask] = 0.09
                
        return xs
    
    def _create_scatter_xs(self, moderator=None):
        """Create energy-dependent scattering cross section."""
        if moderator:
            if moderator == 'H2O':
                xs = 103.0 * np.sqrt(0.025/self.energy_grid)
            elif moderator == 'D2O':
                xs = 13.6 * np.sqrt(0.025/self.energy_grid)
            elif moderator == 'C':
                xs = 4.8 * np.sqrt(0.025/self.energy_grid)
        else:
            # U-235/U-238 scattering cross section
            xs = np.zeros_like(self.energy_grid)
            # Thermal region
            thermal_mask = self.energy_grid <= 0.025
            xs[thermal_mask] = 15.0
            # Resonance region
            resonance_mask = (self.energy_grid > 0.025) & (self.energy_grid <= 1000)
            xs[resonance_mask] = 12.0
            # Fast region
            fast_mask = self.energy_grid > 1000
            xs[fast_mask] = 4.0
            
        return xs
    
    def _create_scattering_law(self, moderator=None):
        """Create energy-dependent scattering law parameters."""
        if moderator:
            if moderator == 'H2O':
                xi = 0.927 * np.ones_like(self.energy_grid)
            elif moderator == 'D2O':
                xi = 0.510 * np.ones_like(self.energy_grid)
            elif moderator == 'C':
                xi = 0.158 * np.ones_like(self.energy_grid)
        else:
            # U-235/U-238 scattering law
            xi = 0.008 * np.ones_like(self.energy_grid)
            
        return xi
    
    def _create_fission_spectrum(self, u238=False):
        """Create fission neutron energy spectrum."""
        if u238:
            # U-238 fast fission spectrum (simplified)
            E = np.linspace(0, 10e6, 1000)  # 0 to 10 MeV
            spectrum = np.exp(-E/2e6)  # Maxwellian-like distribution
        else:
            # U-235 Watt spectrum
            E = np.linspace(0, 10e6, 1000)  # 0 to 10 MeV
            a = 0.988e6  # eV
            b = 2.249e-6  # eV^-1
            spectrum = np.exp(-E/a) * np.sinh(np.sqrt(b*E))
            
        # Normalize spectrum
        spectrum /= np.sum(spectrum)
        return spectrum 