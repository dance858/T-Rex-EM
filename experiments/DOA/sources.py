from enum import Enum
from abc import ABC, abstractmethod
import copy
import warnings
import numpy as np
from conversion import convert_angles

# Taken from https://github.com/morriswmz/doatools.py, which is licensed
# under the MIT license. All credit goes to the original authors.

def _validate_sensor_location_ndim(sensor_locations):
    if sensor_locations.shape[1] < 1 or sensor_locations.shape[1] > 3:
        raise ValueError('Sensor locations can only consists of 1D, 2D or 3D coordinates.')

class SourcePlacement(ABC):
    """Represents the placement of several sources.
    
    This the base abstract class and should not be directly instantiated.
    """

    def __init__(self, locations, units):
        self._locations = locations
        self._units = units

    def __len__(self):
        """Returns the number of sources."""
        return self._locations.shape[0]

    def __getitem__(self, key):
        """Accesses a specific source location or obtains a subset of source
        placement via slicing.

        For instance, ``sources[i]`` and ``sources.locations[i]`` are
        equivalent, and ``sources[:]`` will return a full copy.

        Args:
            key : An integer, slice, or 1D numpy array of indices/boolean masks.

        Notes:
            This is a generic implementation. When ``key`` is a scalar, key is
            treated as an index and normal indexing operation follows. When
            ``key`` is not a scalar, we need to return a new instance with
            source locations specified by ``key``. First, a shallow copy is
            made with :meth:`~copy.copy`. Then the shallow copy's
            location data are set to the source locations specified by
            ``key``. Finally, the shallow copy is returned.
        """
        if np.isscalar(key):
            return self._locations[key]
        if isinstance(key, slice):
            # Slicing results a view. We force a copy here.
            locations = self._locations[key].copy()
        elif isinstance(key, list):
            locations = self._locations[key]
        elif isinstance(key, np.ndarray):
            if key.ndim != 1:
                raise ValueError('1D array expected.')
            locations = self._locations[key]
        else:
            raise KeyError('Unsupported index.')
        new_copy = copy.copy(self)
        new_copy._locations = locations
        return new_copy

    @property
    def size(self):
        """Retrieves the number of sources."""
        return len(self)

    @property
    def locations(self):
        """Retrieves the source locations.
        
        While this property provides read/write access to the underlying ndarray
        storing the source locations. Modifying the underlying ndarray is
        discourage because modified values are not checked for validity.
        """
        return self._locations
    
    @property
    @abstractmethod
    def is_far_field(self):
        """Retrieves whether the source placement is considered far-field.
        
        A far-field source's distance to a sensor array is defined to be
        infinity.
        """
        raise NotImplementedError()

    @property
    def units(self):
        """Retrieves a tuple consisting of units used for each dimension."""
        return self._units

    @property
    @abstractmethod
    def valid_ranges(self):
        """Retrieves the valid ranges for each dimension.

        Returns:
            ((min_1, max_1), ...): A tuple of 2-element tuples of min-max pairs.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def as_unit(self, new_unit):
        """Creates a copy with the source locations converted to the new unit."""
        raise NotImplementedError()

    @abstractmethod
    def calc_spherical_coords(self, ref_locations):
        """Calculates the spherical coordinates relative to reference locations.

        Args:
            ref_locations (~numpy.ndarray): An M x D matrix storing the
                Cartesian coordinates (measured in meters), where M is the
                number of reference locations, and D is the number of dimensions
                of the coordinates (1, 2, or 3).
        
        Returns:
            tuple: A tuple of three M by K matrices containing the ranges, the
            azimuth angles and the elevation angles, respectively. The (m,k)-th
            elements from the three matrices form the spherical coordinates for
            the k-th source relative to the m-th reference location.
        """
        raise NotImplementedError()

    @abstractmethod
    def phase_delay_matrix(self, sensor_locations, wavelength, derivatives=False):
        """Computes the phase delay matrix.

        The phase delay matrix D is an m x k matrix, where D[i,j] is the
        relative phase difference between the i-th sensor and the j-th source
        (usually using the first sensor as the reference). By convention,
        a positive value means that the corresponding signal arrives earlier
        than the referenced one.

        Args:
            sensor_locations: An m x d (d = 1, 2, 3) matrix representing the
                sensor locations using the Cartesian coordinate system.
            wavelength: Wavelength of the carrier wave.
            derivatives: If set to true, also outputs the derivative matrix (or
                matrices) with respect to the source locations. Default value
                is False.
        
        Returns:
            * When ``derivatives`` is ``False``, returns the steering matrix.
            * When ``derivatives`` is ``True``, returns both the steering matrix
              and the derivative matrix (or matrices if possible) with a tuple.

        Notes:
            The phase delay matrix is used in constructing the steering
            matrix. This method is decoupled from the steering matrix method
            because the phase delays are calculated differently for different
            types of sources (e.g. far-field vs. near-field).
        """
        pass

class FarField1DSourcePlacement(SourcePlacement):
    """Creates a far-field 1D source placement.

    Far-field 1D sources are placed within the xy-plane and represented by
    the angles relative to the y-axis (broadside angles for an array placed
    along the x-axis).

    ::

                 y   
                 ^
                 |   /
                 |  /
                 |-/
                 |/
        ---------+---------> x

    Args:
        locations: A list or 1D numpy array representing the source locations.
        unit (str): Can be ``'rad'``, ``'deg'`` or ``'sin'``. ``'sin'`` is a
            special unit where sine value of the broadside angle is used instead
            of the broadside angle itself. Default value is ``'rad'``.
    """

    VALID_RANGES = {
        'rad': (-np.pi/2, np.pi/2),
        'deg': (-90.0, 90.0),
        'sin': (-1.0, 1.0)
    }

    def __init__(self, locations, unit='rad'):
        if isinstance(locations, list):
            locations = np.array(locations)
        if locations.ndim > 1:
            raise ValueError('1D numpy array expected.')
        if unit not in FarField1DSourcePlacement.VALID_RANGES:
            raise ValueError(
                'Unit can only be one of the following: {0}.'
                .format(', '.join(FarField1DSourcePlacement.VALID_RANGES.keys()))
            )
        lb, ub = FarField1DSourcePlacement.VALID_RANGES[unit]
        if np.any(locations < lb) or np.any(locations > ub):
            raise ValueError(
                "When unit is '{0}', source locations must be within [{0}, {1}]."
                .format_map(unit, lb, ub)
            )
        super().__init__(locations, (unit,))

    @staticmethod
    def from_z(z, wavelength, d0, unit='rad'):
        """Creates a far-field 1D source placement from complex roots.
        
        Used in rooting based DOA estimators such as root-MUSIC and ESPRIT.

        Args:
            z: A ndarray of complex roots.
            wavelength (float): Wavelength of the carrier wave.
            d0 (float): Inter-element spacing of the uniform linear array.
            unit (str): Can be ``'rad'``, ``'deg'`` or ``'sin'``. Default value
                is ``'rad'``.
        
        Returns:
            An instance of
            :class:`~doatools.model.sources.FarField1DSourcePlacement`.
        """
        c = 2 * np.pi * d0 / wavelength
        sin_vals = np.angle(z) / c
        if unit == 'sin':
            sin_vals.sort()
            return FarField1DSourcePlacement(sin_vals, 'sin')
        locations = np.arcsin(sin_vals)
        locations.sort()        
        if unit == 'rad':
            return FarField1DSourcePlacement(locations)
        else:
            return FarField1DSourcePlacement(np.rad2deg(locations), 'deg')

    @property
    def is_far_field(self):
        return True

    @property
    def valid_ranges(self):
        return FarField1DSourcePlacement.VALID_RANGES[self._units[0]],

    def as_unit(self, new_unit):
        return FarField1DSourcePlacement(
            convert_angles(self._locations, self._units[0], new_unit),
            new_unit
        )

    def calc_spherical_coords(self, ref_locations):
        m = ref_locations.shape[0]
        k = self.size
        r = np.full((m, k), np.inf)
        el = np.zeros((m, k))
        # Broadside angles are defined relative to the y-axis
        az = np.pi/2 - convert_angles(self.locations, self.units[0], 'rad')
        az = np.tile(az, (m, 1))
        return r, az, el

    def phase_delay_matrix(self, sensor_locations, wavelength, derivatives=False):
        """Computes the phase delay matrix for 1D far-field sources."""
        _validate_sensor_location_ndim(sensor_locations)
        
        if self._units[0] == 'sin':
            return self._phase_delay_matrix_sin(sensor_locations, wavelength, derivatives)
        else:
            return self._phase_delay_matrix_rad(sensor_locations, wavelength, derivatives)
        
    def _phase_delay_matrix_rad(self, sensor_locations, wavelength, derivatives=False):
        # Unit can only be 'rad' or 'deg'.
        # Unify to radians.
        if self._units[0] == 'deg':
            locations = np.deg2rad(self._locations)
        else:
            locations = self._locations
        
        locations = locations[np.newaxis]
        s = 2 * np.pi / wavelength
        if sensor_locations.shape[1] == 1:
            # D[i,k] = sensor_location[i] * sin(doa[k])
            D = s * np.outer(sensor_locations, np.sin(locations))
            if derivatives:
                DD = s * np.outer(sensor_locations, np.cos(locations))
        else:
            # The sources are assumed to be within the xy-plane. The offset
            # along the z-axis of the sensors does not affect the delays.
            # D[i,k] = sensor_location_x[i] * sin(doa[k])
            #          + sensor_location_y[i] * cos(doa[k])
            D = s * (np.outer(sensor_locations[:, 0], np.sin(locations)) +
                     np.outer(sensor_locations[:, 1], np.cos(locations)))
            if derivatives:
                DD = s * (np.outer(sensor_locations[:, 0], np.cos(locations)) -
                          np.outer(sensor_locations[:, 1], np.sin(locations)))
        if self._units[0] == 'deg' and derivatives:
            DD *= np.pi / 180.0 # Do not forget the scaling when unit is 'deg'.
        return (D, DD) if derivatives else D

    def _phase_delay_matrix_sin(self, sensor_locations, wavelength, derivatives=False):
        sin_vals = self._locations
        s = 2 * np.pi / wavelength
        if sensor_locations.shape[1] == 1:
            # D[i,k] = sensor_location[i] * sin_val[k]
            D = s * (sensor_locations * sin_vals)
            if derivatives:
                # Note that if x = \sin\theta then
                # \frac{\partial cx}{\partial x} = c
                # This is different from the derivative w.r.t. \theta:
                # \frac{\partial cx}{\partial \theta} = c\cos\theta
                DD = np.tile(s * sensor_locations, (1, self._locations.size))
        else:
            # The sources are assumed to be within the xy-plane. The offset
            # along the z-axis of the sensors does not affect the delays.
            cos_vals = np.sqrt(1.0 - sin_vals * sin_vals)
            D = s * (np.outer(sensor_locations[:, 0], sin_vals) +
                     np.outer(sensor_locations[:, 1], cos_vals))
            if derivatives:
                # If x = \sin\theta, \theta \in (-\pi/2, \pi/2)
                # a \sin\theta + b \cos\theta = ax + b\sqrt{1-x^2}
                # d/dx(ax + b\sqrt{1-x^2}) = a - bx/\sqrt{1-x^2}
                # sensor_locations[:, 0, np.newaxis] will be a column
                # vector and broadcasting will be utilized.
                DD = s * (sensor_locations[:, 0, np.newaxis] -
                          np.outer(sensor_locations[:, 1], sin_vals / cos_vals))
        return (D, DD) if derivatives else D