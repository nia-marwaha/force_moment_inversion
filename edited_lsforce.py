import copy
import glob
import os
import pickle
import random as rnd
import subprocess
import tempfile
import warnings
from shutil import copyfile, which
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from obspy import Stream, Trace, UTCDateTime, read
from obspy.core import AttribDict
from scipy.signal.windows import triang

# Base URL for Syngine queries
SYNGINE_BASE_URL = 'https://service.iris.edu/irisws/syngine/1/query?'

# [s] Sampling interval for Green's functions downloaded from Syngine
SYNGINE_DT = 0.25

# [s] Sampling interval for the triangular source time function given to Syngine
TRIANGLE_STF_DT = 0.5

# A nice constant start time for Syngine and CPS GFs
GF_STARTTIME = UTCDateTime(1900, 1, 1)

# Convert m/s to μm/s
UMS_PER_MS = 1e6

# Transparency level for jackknife lines and patches
JK_ALPHA = 0.2

# Constants controlling behavior of `plot_fits()`
AX_WIDTH = 2  # [in] Waveform subplot panel width
AX_HEIGHT = 0.5  # [in] Waveform subplot panel height
PAD = 0.1  # [in] Internal padding around waveform subplot panels
MARGINS = (0.75, PAD, 0.2, PAD)  # [in] Outer [left, right, top, bottom] overall margins
COMPONENT_ORDER = ('Z', 'R', 'T')  # Order from left to right of component columns
TEXT_PAD = PAD  # [in]  Padding for station info/metrics labels on left of waveforms
FONT_SIZE = 8  # Main plot font size
PRECISION = 0  # Number of decimal places for station metrics


class LSForce:
    r"""Class for performing force inversions.

    Attributes:
        gf_dir (str): Directory containing Green's functions
        gf_computed (bool): Whether or not Green's functions have been computed for this
            object
        filtered_gf_st (:class:`~obspy.core.stream.Stream`): Stream containing filtered
            Green's functions
        inversion_complete (bool): Whether or not the inversion has been run
        filter (dict): Dictionary with keys ``'freqmin'``, ``'freqmax'``,
            ``'zerophase'``, ``'periodmin'``, ``'periodmax'``, and ``'order'``
            specifying filter parameters
        data_length (int): Length in samples of each data trace
        force_sampling_rate (int or float): [Hz] The sampling rate of the force-time
            function
        W (2D array): Weight matrix
        Wvec (1D array): Weight vector
        jackknife (:class:`~obspy.core.util.attribdict.AttribDict`): Dictionary with
            keys ``'Z'``, ``'N'``, ``'E'``, ``'VR_all'``, ``'alphas'``, ``'num_iter'``,
            ``'frac_delete'``, ``'tr_id_lists_all'``, and ``'type'`` containing
            jackknife parameters and results
        angle_magnitude (:class:`~obspy.core.util.attribdict.AttribDict`): Dictionary
            with keys ``'magnitude'``, ``'magnitude_upper'``, ``'magnitude_lower'``,
            ``'vertical_angle'``, and ``'horizontal_angle'`` containing inversion angle
            and magnitude information
        G (2D array): Design matrix
        d (1D array): Data vector
        model: Model vector of concatenated components (n x 1) of solution
        Z: [N] Vertical force time series extracted from model (positive up)
        N: [N] North force time series extracted from model (positive north)
        E: [N] East force time series extracted from model (positive east)
        tvec: [s] Time vector for forces, referenced using `zero_time` (if specified)
        VR: [%] Variance reduction. Rule of thumb: This should be ~50–80%, if ~100%,
            solution is fitting data exactly and results are suspect. If ~5%, model may
            be wrong or something else may be wrong with setup
        dtorig: Original data vector
        dtnew: Modeled data vector (Gm-d)
        alpha: Regularization parameter that was used
        alphafit (dict): Dictionary with keys ``'alphas'``, ``'fit'``, and ``'size'``
            specifying regularization parameters tested
    """

    def __init__(self, data, data_sampling_rate, main_folder=None, method='full'):
        r"""Create an LSForce object.

        Args:
            data (:class:`~lsforce.lsdata.LSData`): LSData object, corrected for station
                response but not filtered
            data_sampling_rate (int or float): [Hz] Samples per second to use in
                inversion. All data will be resampled to this rate, and Green's
                functions will be created with this rate
            main_folder (str): If `None`, will use current folder
            method (str): How to parameterize the force-time function. One of `'full'`
                — full inversion using Tikhonov regularization (L2 norm minimization) or
                `'triangle'` — inversion parameterized using overlapping triangles,
                variation on method of Ekström & Stark (2013)
        """
        # Check that the data are sorted as expected
        _st_proc = data.st_proc.copy()
        _st_proc.sort(keys=['distance', 'station', 'channel'])
        assert [tr.id for tr in _st_proc] == [tr.id for tr in data.st_proc]

        self.data = data
        self.data_sampling_rate = data_sampling_rate
        self.gf_computed = False
        self.inversion_complete = False

        if not main_folder:
            self.main_folder = os.getcwd()
        else:
            self.main_folder = main_folder

        if method not in ['full', 'triangle']:
            raise ValueError(f'Method {method} not yet implemented.')

        self.method = method

    def _get_greens(self):
        r"""Get the Green's function Stream using either Syngine or CPS."""

        # Name the directory where GFs will be stored based on method
        if self.cps_model:
            gf_dir_name = f'cps_{os.path.basename(self.cps_model).split(".")[-2]}'
        else:
            gf_dir_name = f'syngine_{self.syngine_model}'
        self.gf_dir = os.path.join(self.main_folder, gf_dir_name)

        # Label directories containing triangular GFs as such
        if self.method == 'triangle':
            self.gf_dir += f'_triangle_{self.triangle_half_width:g}s'

        # Make GF directory if it doesn't exist
        if not os.path.exists(self.gf_dir):
            os.mkdir(self.gf_dir)

        # Choose the correct delta
        if self.cps_model:
            gf_dt = 1 / self.data_sampling_rate
        else:
            gf_dt = SYNGINE_DT

        # Get list of unique stations
        unique_stations = np.unique([tr.stats.station for tr in self.data.st_proc])

        # Make lists of stations with and without GFs calculated/downloaded
        existing_stations = []
        stations_to_calculate = []
        for station in unique_stations:
            distance = self.data.st_proc.select(station=station)[0].stats.distance
            filename = os.path.join(self.gf_dir, f'{station}.pkl')

            # Check if this EXACT GF exists already
            if os.path.exists(filename):
                stats = read(filename)[0].stats
                if (
                    stats.syngine_model == self.syngine_model
                    and stats.cps_model == self.cps_model
                    and stats.triangle_half_width == self.triangle_half_width
                    and stats.sourcedepthinmeters == self.source_depth
                    and stats.distance == distance
                    and stats.T0 == self.T0
                    and stats.duration == self.gf_duration
                    and stats.delta == gf_dt
                ):
                    gf_exists = True
                else:
                    gf_exists = False
            else:
                gf_exists = False

            # Append to the correct vector
            if gf_exists:
                existing_stations.append(station)
            else:
                stations_to_calculate.append(station)

        # Initialize empty Stream to hold all GFs
        st_gf = Stream()

        # CPS
        if self.cps_model:
            # If we have to calculate some stations, go through the process
            if stations_to_calculate:
                # Print status
                print(
                    f'Calculating Green\'s functions for {len(stations_to_calculate)} '
                    'station(s):'
                )
                for station in stations_to_calculate:
                    print(f'\t{station}')

                # Create a temporary directory to run CPS in, and change into it
                cwd = os.getcwd()
                temp_dir = tempfile.TemporaryDirectory()
                os.chdir(temp_dir.name)

                # Copy the model file over to temporary directory to avoid model paths too long for CPS
                modelfilename = os.path.basename(
                    self.cps_model
                )  # get just model filename without path
                copyfile(self.cps_model, os.path.join(temp_dir.name, modelfilename))

                # Write the "dist" file
                gf_length_samples = int(self.gf_duration * self.data_sampling_rate)
                with open('dist', 'w') as f:
                    for sta in stations_to_calculate:
                        dist = self.data.st_proc.select(station=sta)[0].stats.distance
                        f.write(f'{dist} {gf_dt} {gf_length_samples} {self.T0} 0\n')
                # # move copy of dist to main folder for debugging
                # copyfile(os.path.join(temp_dir.name, 'dist'), os.path.join(self.main_folder, 'dist'))

                # Run hprep96 and hspec96
                cpscall = [
                    'hprep96',
                    '-HR',
                    '0.',
                    '-HS',
                    str(self.source_depth / 1000),  # Converting to km for CPS
                    '-M',
                    modelfilename,
                    '-d',
                    'dist',
                    '-R',
                    '-EXF',
                ]
                # print('Running: %s' % subprocess.list2cmdline(cpscall))
                subprocess.call(cpscall)
                with open('hspec96.out', 'w') as f:
                    subprocess.call('hspec96', stdout=f)
                # Run hpulse96 (using the multiplier here to get a 1 N impulse), also
                # keep track of pulse half-width so we can make the GFs acausal later
                args = ['hpulse96', '-d', 'dist', '-m', f'{1e-10:.10f}', '-OD', '-V']
                if self.triangle_half_width is not None:
                    args += ['-t', '-l', str(int(self.triangle_half_width / gf_dt))]
                    pulse_half_width = self.triangle_half_width  # [s]
                else:
                    args += ['-p']
                    pulse_half_width = 2 * gf_dt  # [s]
                with open('Green', 'w') as f:
                    subprocess.call(args, stdout=f)

                # Convert to SAC files
                subprocess.call(['f96tosac', '-B', 'Green'])

                # Go through and read in files (same order as dist file)
                for i, station in enumerate(stations_to_calculate):
                    for file in glob.glob(f'B{i + 1:03d}1???F.sac'):
                        # Grab stats of data trace
                        stats = self.data.st_proc.select(station=station)[0].stats

                        # Read GF in as an ObsPy Trace
                        gf_tr = read(file)[0]

                        # Add metadata
                        gf_tr.stats.network = stats.network
                        gf_tr.stats.station = station
                        gf_tr.stats.location = 'SE'
                        gf_tr.stats.distance = stats.distance
                        gf_tr.stats.syngine_model = self.syngine_model
                        gf_tr.stats.cps_model = self.cps_model
                        gf_tr.stats.sourcedepthinmeters = self.source_depth
                        gf_tr.stats.T0 = self.T0
                        gf_tr.stats.duration = self.gf_duration
                        gf_tr.stats.triangle_half_width = self.triangle_half_width

                        gf_tr.data *= 0.01  # Convert from cm to m

                        # Add Trace to overall GF Stream
                        st_gf += gf_tr

                # Clean up
                temp_dir.cleanup()
                os.chdir(cwd)

                # Trim to length (gf_duration - T0) seconds, and correct for the pulse
                # half-width to make GFs acausal (since the -Z flag doesn't work!)
                starttime = st_gf[0].stats.starttime
                endtime = starttime + self.gf_duration - self.T0
                st_gf.trim(starttime + pulse_half_width, endtime + pulse_half_width)

                # Give a nice start time
                for tr in st_gf:
                    tr.stats.starttime = GF_STARTTIME

                # Save as individual files
                for station in stations_to_calculate:
                    filename = os.path.join(self.gf_dir, f'{station}.pkl')
                    st_gf.select(station=station).write(filename, format='PICKLE')

            # Now just load in the GFs which already exist
            for i, station in enumerate(existing_stations):
                filename = os.path.join(self.gf_dir, f'{station}.pkl')
                st_gf += read(filename)

                # Print status
                progress = len(stations_to_calculate) + i + 1
                print(f'Found {station} ({progress}/{len(unique_stations)})')

        # Syngine
        else:
            # Go station-by-station
            for i, station in enumerate(unique_stations):
                filename = os.path.join(self.gf_dir, f'{station}.pkl')

                # Either load this GF if it already exists, or download it
                if station in existing_stations:
                    st_syn = read(filename)
                else:
                    # Grab stats of data trace
                    stats = self.data.st_proc.select(station=station)[0].stats

                    # Get GFs for this station
                    st_syn = self._get_greens_for_station(
                        receiverlatitude=stats.latitude,
                        receiverlongitude=stats.longitude,
                        back_azimuth=stats.back_azimuth,
                        distance=stats.distance,
                    )

                    # Add metadata, as Syngine uses 'XX' and 'SYN' by default
                    for tr in st_syn:
                        tr.stats.network = stats.network
                        tr.stats.station = stats.station

                    # TODO: Understand why we have to flip the polarity of these!
                    for channel in 'RHF', 'RVF', 'THF':
                        for tr in st_syn.select(channel=channel):
                            tr.data *= -1

                    st_syn.write(filename, format='PICKLE')

                # Add this station's GFs to overall Stream
                st_gf += st_syn

                # Print status
                if station in existing_stations:
                    action_string = 'Found'
                else:
                    action_string = 'Downloaded'
                print(f'{action_string} {station} ({i + 1}/{len(unique_stations)})')

        self.gf_computed = True

        return st_gf

    def _get_greens_for_station(
        self, receiverlatitude, receiverlongitude, back_azimuth, distance
    ):
        r"""Get the Green's functions for a single station (Syngine)."""

        # Provide triangle STF params if we're using the triangle method
        if self.method == 'triangle':
            stf_offset = self.triangle_half_width  # Ensure peak of triangle at t=0
            stf_spacing = TRIANGLE_STF_DT
            # The below construction ensures the triangle is centered on 1 and goes to 0
            # at each end, e.g. [0, 0.5, 1, 0.5, 0] instead of [0.25, 0.75, 0.75, 0.25]
            stf_data = np.hstack(
                [
                    0,
                    triang((int(self.triangle_half_width / TRIANGLE_STF_DT) * 2) - 1),
                    0,
                ]
            )
            read_func = _read  # Use the long-URL wrapper for ObsPy read
        else:
            stf_offset = None
            stf_spacing = None
            stf_data = None
            read_func = read  # Just directly read using ObsPy

        # Convert to radians for NumPy
        back_azimuth_radians = np.deg2rad(back_azimuth)

        # Vertical force (downward)
        st_vf = read_func(
            self._build_syngine_url(
                receiverlatitude=receiverlatitude,
                receiverlongitude=receiverlongitude,
                components='ZR',
                forces=(-1, 0, 0),
                stf_offset=stf_offset,
                stf_spacing=stf_spacing,
                stf_data=stf_data,
            )
        )
        for tr in st_vf.select(component='Z'):
            tr.stats.channel = 'ZVF'
        for tr in st_vf.select(component='R'):
            tr.stats.channel = 'RVF'

        # Horizontal force (radial)
        st_hf_r = read_func(
            self._build_syngine_url(
                receiverlatitude=receiverlatitude,
                receiverlongitude=receiverlongitude,
                components='ZR',
                forces=(0, np.cos(back_azimuth_radians), -np.sin(back_azimuth_radians)),
                stf_offset=stf_offset,
                stf_spacing=stf_spacing,
                stf_data=stf_data,
            )
        )
        for tr in st_hf_r.select(component='Z'):
            tr.stats.channel = 'ZHF'
        for tr in st_hf_r.select(component='R'):
            tr.stats.channel = 'RHF'

        # Horizontal force (transverse)
        st_hf_t = read_func(
            self._build_syngine_url(
                receiverlatitude=receiverlatitude,
                receiverlongitude=receiverlongitude,
                components='T',
                forces=(
                    0,
                    -np.sin(back_azimuth_radians),
                    -np.cos(back_azimuth_radians),
                ),
                stf_offset=stf_offset,
                stf_spacing=stf_spacing,
                stf_data=stf_data,
            )
        )
        for tr in st_hf_t.select(component='T'):
            tr.stats.channel = 'THF'

        # Assemble big Stream
        st_syn = st_vf + st_hf_r + st_hf_t

        # Add metadata and sort
        for tr in st_syn:
            tr.stats.distance = distance
            tr.stats.cps_model = self.cps_model
            tr.stats.syngine_model = self.syngine_model
            tr.stats.sourcedepthinmeters = self.source_depth
            tr.stats.T0 = self.T0
            tr.stats.duration = self.gf_duration
            tr.stats.triangle_half_width = self.triangle_half_width
            tr.stats.starttime = GF_STARTTIME  # Give a nice start time
        st_syn.sort(keys=['channel'])

        return st_syn

    def _build_syngine_url(
        self,
        receiverlatitude,
        receiverlongitude,
        components,
        forces,
        stf_offset=None,
        stf_spacing=None,
        stf_data=None,
    ):
        r"""Build a URL to be fed to Syngine."""

        parameters = [
            'format=miniseed',
            'components=' + components,
            'units=displacement',
            'model=' + self.syngine_model,
            'dt=' + str(SYNGINE_DT),
            'starttime=' + str(self.T0),
            'endtime=' + str(self.gf_duration - self.T0),
            'receiverlatitude=' + str(receiverlatitude),
            'receiverlongitude=' + str(receiverlongitude),
            'sourcelatitude=' + str(self.data.source_lat),
            'sourcelongitude=' + str(self.data.source_lon),
            'sourcedepthinmeters=' + str(self.source_depth),
            'sourceforce=' + ','.join([str(force) for force in forces]),
            'nodata=404',
        ]

        if stf_offset is not None and stf_spacing is not None and stf_data is not None:
            parameters += [
                'cstf-relative-origin-time-in-sec=' + str(stf_offset),
                'cstf-sample-spacing-in-sec=' + str(stf_spacing),
                'cstf-data=' + ','.join([str(sample) for sample in stf_data]),
            ]
        elif stf_offset is not None or stf_spacing is not None or stf_data is not None:
            raise ValueError('All three CSTF parameters must be provided!')

        url = SYNGINE_BASE_URL + '&'.join(parameters)

        return url

    def _get_greens_mt(self):
        r"""Get the moment tensor Green's function Stream using Syngine."""

        if self.cps_model:
            gf_dir_name = f'cps_{os.path.basename(self.cps_model).split(".")[-2]}_mt'
        else:
            gf_dir_name = f'syngine_{self.syngine_model}_mt'
        self.gf_dir = os.path.join(self.main_folder, gf_dir_name)

        if self.method == 'triangle':
            self.gf_dir += f'_triangle_{self.triangle_half_width:g}s'

        if not os.path.exists(self.gf_dir):
            os.mkdir(self.gf_dir)

        if self.cps_model:
            gf_dt = 1 / self.data_sampling_rate
        else:
            gf_dt = SYNGINE_DT

        unique_stations = np.unique([tr.stats.station for tr in self.data.st_proc])

        existing_stations = []
        stations_to_calculate = []
        for station in unique_stations:
            distance = self.data.st_proc.select(station=station)[0].stats.distance
            filename = os.path.join(self.gf_dir, f'{station}.pkl')

            if os.path.exists(filename):
                stats = read(filename)[0].stats
                if (
                    stats.syngine_model == self.syngine_model
                    and stats.cps_model == self.cps_model
                    and stats.triangle_half_width == self.triangle_half_width
                    and stats.sourcedepthinmeters == self.source_depth
                    and stats.distance == distance
                    and stats.T0 == self.T0
                    and stats.duration == self.gf_duration
                    and stats.delta == gf_dt
                ):
                    gf_exists = True
                else:
                    gf_exists = False
            else:
                gf_exists = False

            if gf_exists:
                existing_stations.append(station)
            else:
                stations_to_calculate.append(station)

        st_gf = Stream()

        for i, station in enumerate(unique_stations):
            filename = os.path.join(self.gf_dir, f'{station}.pkl')

            if station in existing_stations:
                st_syn = read(filename)
            else:
                stats = self.data.st_proc.select(station=station)[0].stats

                st_syn = self._get_greens_for_station_mt(
                    receiverlatitude=stats.latitude,
                    receiverlongitude=stats.longitude,
                    back_azimuth=stats.back_azimuth,
                    distance=stats.distance,
                )

                for tr in st_syn:
                    tr.stats.network = stats.network
                    tr.stats.station = stats.station

                st_syn.write(filename, format='PICKLE')

            st_gf += st_syn

            action_string = 'Found' if station in existing_stations else 'Downloaded'
            print(f'{action_string} {station} ({i + 1}/{len(unique_stations)})')

        self.gf_computed = True

        return st_gf

    def _get_greens_for_station_mt(self, receiverlatitude, receiverlongitude, back_azimuth, distance):
        r"""Get the 6 fundamental moment tensor Green's functions for a single
        station.
        """

        if self.method == 'triangle':
            stf_offset = self.triangle_half_width
            stf_spacing = TRIANGLE_STF_DT
            stf_data = np.hstack(
                [
                    0,
                    triang((int(self.triangle_half_width / TRIANGLE_STF_DT) * 2) - 1),
                    0,
                ]
            )
            read_func = _read
        else:
            stf_offset = None
            stf_spacing = None
            stf_data = None
            read_func = read

        # Order matches Syngine's sourcemomenttensor=Mrr,Mtt,Mpp,Mrt,Mrp,Mtp
        mt_elements = ['RR', 'TT', 'PP', 'RT', 'RP', 'TP']

        st_syn = Stream()
        for idx, element in enumerate(mt_elements):
            moment_tensor = [0, 0, 0, 0, 0, 0]
            moment_tensor[idx] = 1  

            st_element = read_func(
                self._build_syngine_url_mt(
                    receiverlatitude=receiverlatitude,
                    receiverlongitude=receiverlongitude,
                    components='ZRT',
                    moment_tensor=moment_tensor,
                    stf_offset=stf_offset,
                    stf_spacing=stf_spacing,
                    stf_data=stf_data,
                )
            )
            for tr in st_element:
                tr.stats.channel = tr.stats.channel[-1] + element 

            st_syn += st_element

        for tr in st_syn:
            tr.stats.distance = distance
            tr.stats.back_azimuth = back_azimuth
            tr.stats.cps_model = self.cps_model
            tr.stats.syngine_model = self.syngine_model
            tr.stats.sourcedepthinmeters = self.source_depth
            tr.stats.T0 = self.T0
            tr.stats.duration = self.gf_duration
            tr.stats.triangle_half_width = self.triangle_half_width
            tr.stats.starttime = GF_STARTTIME
        st_syn.sort(keys=['channel'])

        return st_syn


    def _build_syngine_url_mt(
        self,
        receiverlatitude,
        receiverlongitude,
        components,
        moment_tensor,
        stf_offset=None,
        stf_spacing=None,
        stf_data=None,
    ):
        r"""Build a Syngine URL for a moment tensor source."""

        parameters = [
            'format=miniseed',
            'components=' + components,
            'units=displacement',
            'model=' + self.syngine_model,
            'dt=' + str(SYNGINE_DT),
            'starttime=' + str(self.T0),
            'endtime=' + str(self.gf_duration - self.T0),
            'receiverlatitude=' + str(receiverlatitude),
            'receiverlongitude=' + str(receiverlongitude),
            'sourcelatitude=' + str(self.data.source_lat),
            'sourcelongitude=' + str(self.data.source_lon),
            'sourcedepthinmeters=' + str(self.source_depth),
            'sourcemomenttensor=' + ','.join([str(m) for m in moment_tensor]),
            'nodata=404',
        ]

        if stf_offset is not None and stf_spacing is not None and stf_data is not None:
            parameters += [
                'cstf-relative-origin-time-in-sec=' + str(stf_offset),
                'cstf-sample-spacing-in-sec=' + str(stf_spacing),
                'cstf-data=' + ','.join([str(sample) for sample in stf_data]),
            ]
        elif stf_offset is not None or stf_spacing is not None or stf_data is not None:
            raise ValueError('All three CSTF parameters must be provided!')

        url = SYNGINE_BASE_URL + '&'.join(parameters)

        return url

    def setup(
        self,
        period_range,
        syngine_model=None,
        cps_model=None,
        triangle_half_width=None,
        source_depth=0,
        weights=None,
        noise_window_dur=None,
        filter_order=2,
        zerophase=True,
        skip_datafilter=False,
        source_type='force'
    ):
        r"""Downloads/computes Green's functions (GFs) and creates all matrices.

        Args:
            period_range (list or tuple): [s] Bandpass filter corners
            syngine_model (str): Name of Syngine model to use. If this is not None, then
                we calculate GFs using Syngine (preferred)
            cps_model (str): Filename of CPS model to use. If this is not None, then we
                calculate GFs using CPS
            triangle_half_width (int or float): [s] Half-width of triangles; only used
                if the triangle method is being used
            source_depth (int or float): [m] Source depth in meters
            weights (list or tuple or str): If `None`, no weighting is applied. An array
                of floats with length ``st_proc.count()`` (and in the order of the ``st_proc``
                attribute of the :class:`~lsforce.lsdata.LSData` object) applies manual
                weighting. If `'prenoise'`, uses standard deviation of a noise window
                defined by `noise_window_dur` to weight. If `'distance'`, weights by 1 /
                distance
            noise_window_dur (int or float): [s] Length of noise window for `'prenoise'`
                weighting scheme (if not `None`, `weights` is set to `'prenoise'`)
            filter_order (int): Order of filter applied over period_range
            zerophase (bool): If `True`, zero-phase filtering will be used
            skip_datafilter (bool): If `True`, filtering will not be applied to
                the input data and will only be applied to the Green's functions.
                This should be chosen only if the data were pre-filtered
                manually already with the same band as `period_range` so the
                user doesn't want to filter them again
        """

        self.syngine_model = syngine_model
        self.source_depth = source_depth
        self.source_type = source_type 

        # Check if input data were pre-filtered and suggest skip_datafilter if not set to True
        if self.data._is_pre_filt and not skip_datafilter:
            warnings.warn(
                'Caution: pre-filtering appears to have been applied'
                ' to input data. Setting `skip_datafilter=True` is'
                ' recommended if you want to avoid double filtering'
            )

        # Explicitly ignore the triangle half-width parameter if it's not relevant
        if self.method != 'triangle' and triangle_half_width is not None:
            triangle_half_width = None
            print(
                'Ignoring `triangle_half_width` parameter since you\'re not using the '
                'triangle method.'
            )

        # Make sure user specifies the triangle half-width if they want that method
        if self.method == 'triangle' and triangle_half_width is None:
            raise ValueError('triangle method is specified but no half-width given!')
        self.triangle_half_width = triangle_half_width

        # If user wants CPS to be run, make sure that 1) they have it installed; 2) it
        # runs; and 3) they have provided a valid filepath
        if cps_model:
            # 1) Is CPS installed?
            if not which('hprep96'):
                raise OSError(
                    'CPS Green\'s function calculation requested, but CPS not found on '
                    'system. Install CPS and try again.'
                )
            # 2) Does CPS run?
            if subprocess.call('hprep96', stderr=subprocess.DEVNULL) != 0:
                raise OSError('Issue with CPS. Check install and try again.')
            # 3) Is `cps_model` a file?
            if not os.path.exists(cps_model):
                raise OSError(f'Could not find CPS model file "{cps_model}"')
            else:
                cps_model = os.path.abspath(cps_model)  # Get full path
        self.cps_model = cps_model

        # The user must specify ONE of `syngine_model` and `cps_model`
        if (self.syngine_model and self.cps_model) or (
            not self.syngine_model and not self.cps_model
        ):
            raise ValueError('You must specify ONE of syngine_model or cps_model!')

        # Automatically choose an appropriate T0 and GF duration based on data/method
        if self.method == 'triangle':
            self.T0 = -2 * self.triangle_half_width  # [s] Double the half-width
        else:
            self.T0 = -10  # [s]
        min_time = np.min([tr.stats.starttime for tr in self.data.st_proc])
        max_time = np.max([tr.stats.endtime for tr in self.data.st_proc])
        self.gf_duration = max_time - min_time  # [s]

        # Create filter dictionary to keep track of filter used without creating too
        # many new attributes
        self.filter = {
            'freqmin': 1.0 / period_range[1],
            'freqmax': 1.0 / period_range[0],
            'zerophase': zerophase,
            'periodmin': period_range[0],
            'periodmax': period_range[1],
            'order': filter_order,
        }

        # Clear weights
        self.Wvec = None
        self.W = None

        if weights is None:
            # Don't weight at all
            weight_method = None
        elif isinstance(weights, str):
            # The user specified a weight method
            weight_method = weights
        else:
            # The user specified a vector of station weights
            weight_method = 'Manual'
            self.weights = weights

        if weight_method != 'Manual':
            if weights == 'prenoise' and noise_window_dur is None:
                raise ValueError(
                    'noise_window_dur must be defined if prenoise weighting is used.'
                )

        # Check if sampling rate specified is compatible with period_range
        if 2.0 * self.filter['freqmax'] > self.data_sampling_rate:
            raise ValueError(
                'data_sampling_rate and period_range are not compatible (violates '
                'Nyquist).'
            )

        # Always work on copy of data
        st = self.data.st_proc.copy()

        # Filter data to band specified
        if not skip_datafilter:
            st.filter(
                'bandpass',
                freqmin=self.filter['freqmin'],
                freqmax=self.filter['freqmax'],
                corners=self.filter['order'],
                zerophase=self.filter['zerophase'],
            )

        # Resample st to data_sampling_rate
        st.resample(self.data_sampling_rate, window='hann')

        # Make sure st data are all the same length
        lens = [len(trace.data) for trace in st]
        if len(set(lens)) != 1:
            print(
                'Resampled records are of differing lengths. Interpolating all records '
                'to same start time and sampling rate.'
            )
            stts = [tr.stats.starttime for tr in st]
            lens = [tr.stats.npts for tr in st]
            st.interpolate(
                self.data_sampling_rate, starttime=np.max(stts), npts=np.min(lens) - 1
            )

        self.data_length = st[0].stats.npts
        total_data_length = self.data_length * st.count()

        # Load in GFs
        print('Getting Green\'s functions...')
        if self.source_type == 'force':
            st_gf = self._get_greens()
        elif self.source_type == 'moment': 
            st_gf = self._get_greens_mt()
        else: 
            st_gf = self._get_greens()+self._get_greens_mt()

        # Process GFs in bulk
        st_gf.detrend()
        st_gf.taper(max_percentage=0.05)
        st_gf.filter(
            'bandpass',
            freqmin=self.filter['freqmin'],
            freqmax=self.filter['freqmax'],
            corners=self.filter['order'],
            zerophase=self.filter['zerophase'],
        )
        if self.syngine_model:  # Only need to do this if Syngine
            st_gf.interpolate(
                sampling_rate=self.data_sampling_rate, method='lanczos', a=20
            )
        st_gf.sort(keys=['channel'])

        # Store the filtered GFs
        self.filtered_gf_st = st_gf

        # Initialize weighting matrices
        Wvec = np.ones(total_data_length)
        indx = 0
        weight = np.ones(self.data.st_proc.count())

        # Store data length
        n = self.data_length

        if self.method == 'full':
            # Set sampling rate
            self.force_sampling_rate = self.data_sampling_rate
        elif self.method == 'triangle':
            self.force_sampling_rate = 1.0 / self.triangle_half_width
            # Number of samples to shift each triangle by
            fshiftby = int(self.triangle_half_width * self.data_sampling_rate)
            # Number of shifts, corresponds to length of force time function
            Flen = int(np.floor(self.data_length / fshiftby))
            # Triangle GFs are multiplied by the triangle half-width so that they
            # reflect the ground motion induced for a triangle with PEAK 1 N instead of
            # AREA of 1 N*s
            for tr in st_gf:
                tr.data = tr.data * self.triangle_half_width
        else:
            raise ValueError(f'Method {self.method} not supported.')

        for i, tr in enumerate(st):
            # Find component and station of Trace
            component = tr.stats.channel[-1]
            station = tr.stats.station


            if self.source_type == 'force':
                if component == 'Z':
                    zvf = st_gf.select(station=station, channel='ZVF')[0]
                    zhf = st_gf.select(station=station, channel='ZHF')[0]
                    if self.method == 'full':
                        ZVF = _makeconvmat(zvf.data, size=(n, n))
                        ZHF = _makeconvmat(zhf.data, size=(n, n))
                    else:
                        ZVF = _makeshiftmat(zvf.data, shiftby=fshiftby, size1=(n, Flen))
                        ZHF = _makeshiftmat(zhf.data, shiftby=fshiftby, size1=(n, Flen))
                    az_radians = np.deg2rad(tr.stats.azimuth)
                    newline = np.hstack(
                        [ZVF, ZHF * np.cos(az_radians), ZHF * np.sin(az_radians)]
                    )

                elif component == 'R':
                    rvf = st_gf.select(station=station, channel='RVF')[0]
                    rhf = st_gf.select(station=station, channel='RHF')[0]
                    if self.method == 'full':
                        RVF = _makeconvmat(rvf.data, size=(n, n))
                        RHF = _makeconvmat(rhf.data, size=(n, n))
                    else:
                        RVF = _makeshiftmat(rvf.data, shiftby=fshiftby, size1=(n, Flen))
                        RHF = _makeshiftmat(rhf.data, shiftby=fshiftby, size1=(n, Flen))
                    az_radians = np.deg2rad(tr.stats.azimuth)
                    newline = np.hstack(
                        [RVF, RHF * np.cos(az_radians), RHF * np.sin(az_radians)]
                    )

                elif component == 'T':
                    thf = st_gf.select(station=station, channel='THF')[0]
                    if self.method == 'full':
                        THF = _makeconvmat(thf.data, size=(n, n))
                    else:
                        THF = _makeshiftmat(thf.data, shiftby=fshiftby, size1=(n, Flen))
                    TVF = 0.0 * THF.copy()
                    az_radians = np.deg2rad(tr.stats.azimuth)
                    newline = np.hstack(
                        [TVF, THF * np.sin(az_radians), -THF * np.cos(az_radians)]
                    )

                else:
                    raise ValueError(f'Data not rotated to ZRT for {station}.')

            elif self.source_type == 'moment':  
                mt_elements = ['RR', 'TT', 'PP', 'RT', 'RP', 'TP']
                cols = []
                for element in mt_elements:
                    gf_tr = st_gf.select(station=station, channel=component + element)[0]
                    if self.method == 'full':
                        col = _makeconvmat(gf_tr.data, size=(n, n))
                    else:
                        col = _makeshiftmat(gf_tr.data, shiftby=fshiftby, size1=(n, Flen))
                    cols.append(col)
                newline = np.hstack(cols)
            
            else:
                mt_elements = ['RR', 'TT', 'PP', 'RT', 'RP', 'TP']

                if component == 'Z':
                    zvf = st_gf.select(station=station, channel='ZVF')[0]
                    zhf = st_gf.select(station=station, channel='ZHF')[0]
                    if self.method == 'full':
                        ZVF = _makeconvmat(zvf.data, size=(n, n))
                        ZHF = _makeconvmat(zhf.data, size=(n, n))
                    else:
                        ZVF = _makeshiftmat(zvf.data, shiftby=fshiftby, size1=(n, Flen))
                        ZHF = _makeshiftmat(zhf.data, shiftby=fshiftby, size1=(n, Flen))
                    az_radians = np.deg2rad(tr.stats.azimuth)
                    force_cols = np.hstack(
                        [ZVF, ZHF * np.cos(az_radians), ZHF * np.sin(az_radians)]
                    )

                elif component == 'R':
                    rvf = st_gf.select(station=station, channel='RVF')[0]
                    rhf = st_gf.select(station=station, channel='RHF')[0]
                    if self.method == 'full':
                        RVF = _makeconvmat(rvf.data, size=(n, n))
                        RHF = _makeconvmat(rhf.data, size=(n, n))
                    else:
                        RVF = _makeshiftmat(rvf.data, shiftby=fshiftby, size1=(n, Flen))
                        RHF = _makeshiftmat(rhf.data, shiftby=fshiftby, size1=(n, Flen))
                    az_radians = np.deg2rad(tr.stats.azimuth)
                    force_cols = np.hstack(
                        [RVF, RHF * np.cos(az_radians), RHF * np.sin(az_radians)]
                    )

                elif component == 'T':
                    thf = st_gf.select(station=station, channel='THF')[0]
                    if self.method == 'full':
                        THF = _makeconvmat(thf.data, size=(n, n))
                    else:
                        THF = _makeshiftmat(thf.data, shiftby=fshiftby, size1=(n, Flen))
                    TVF = 0.0 * THF.copy()
                    az_radians = np.deg2rad(tr.stats.azimuth)
                    force_cols = np.hstack(
                        [TVF, THF * np.sin(az_radians), -THF * np.cos(az_radians)]
                    )


                else:
                    raise ValueError(f'Data not rotated to ZRT for {station}.')
                
                mt_cols_list = []
                for element in mt_elements:
                    gf_tr = st_gf.select(station=station, channel=component + element)[0]
                    if self.method == 'full':
                        col = _makeconvmat(gf_tr.data, size=(n, n))
                    else:
                        col = _makeshiftmat(gf_tr.data, shiftby=fshiftby, size1=(n, Flen))
                    mt_cols_list.append(col)
                mt_cols = np.hstack(mt_cols_list)
                newline=np.hstack((force_cols, mt_cols))

            # Deal with data
            datline = tr.data

            if i == 0:  # If this is the first station, initialize G and d
                G = newline.copy()
                d = datline.copy()
            else:  # Otherwise build on G and d
                G = np.vstack((G, newline.copy()))
                d = np.hstack((d, datline.copy()))

            if weights is not None:
                if weight_method == 'Manual':
                    weight[i] = weights[i]
                elif weights == 'prenoise':
                    weight[i] = 1.0 / np.std(
                        tr.data[0 : int(noise_window_dur * tr.stats.sampling_rate)]
                    )
                elif weights == 'distance':
                    weight[i] = tr.stats.distance

                Wvec[indx : indx + self.data_length] = (
                    Wvec[indx : indx + self.data_length] * weight[i]
                )
                indx += self.data_length
        if self.method == 'full':
            # Need to multiply G by sample interval [s] since convolution is an integral
            self.G = G * 1.0 / self.data_sampling_rate
        else:
            # We don't need to scale the triangle method GFs by the sample rate since
            # this method is not a convolution
            self.G = G

        # Normalize Wvec so largest weight is 1.
        self.Wvec = Wvec / np.max(np.abs(Wvec))
        self.weights = weight / np.max(np.abs(weight))

        if np.shape(G)[0] != len(d):
            raise ValueError(
                'G and d sizes are not compatible, fix something somewhere.'
            )

        self.d = d
        if weights is not None:
            self.W = np.diag(self.Wvec)
        else:
            self.W = None

    def forward_both(self, Z, N, E, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp):
            r"""Execute the forward problem :math:`\mathbf{d} = \mathbf{G}\mathbf{m}`
            using user-supplied force and moment tensor time series, for a
            source_type='both' (joint) Green's function matrix.
    
            Args:
                Z, N, E: [N] Force time series
                Mrr, Mtt, Mpp, Mrt, Mrp, Mtp: [N·m] Moment tensor component time series
    
            Returns:
                :class:`~obspy.core.stream.Stream`: [m] Stream containing synthetic
                data, :math:`\mathbf{d}`
            """
            if not self.gf_computed:
                raise RuntimeError('G is not defined. Did you run LSForce.setup()?')
            if self.source_type != 'both':
                raise RuntimeError(
                    f'forward_both() requires source_type="both", but this object '
                    f'was set up with source_type="{self.source_type}".'
                )
    
            comps = [Z, N, E, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
            comps = [np.array(c).squeeze() for c in comps]
            for vec in comps:
                assert (
                    vec.size == self.data_length
                ), f'Each component must have length {self.data_length}!'
    
            # Z is negated to match the up-positive force convention used in setup()
            model = np.hstack([-comps[0], *comps[1:]])
    
            d = np.reshape(self.G.dot(model), (-1, self.data_length))
    
            st_syn = self.data.st_proc.copy()
            for data, tr in zip(d, st_syn):
                tr.stats.sampling_rate = self.data_sampling_rate
                tr.data = data
                tr.stats.location = 'SE'
                tr.stats.channel = _get_band_code(tr.stats.delta) + 'X' + tr.stats.component
                for key in (
                    '_fdsnws_dataselect_url',
                    '_format',
                    'mseed',
                    'processing',
                    'response',
                ):
                    try:
                        del tr.stats[key]
                    except KeyError:
                        pass
    
            return st_syn

def _varred(dt, dtnew):
    r"""Compute variance reduction :math:`\mathrm{VR}`.

    The formula is

    .. math::

        \mathrm{VR} = \left(1 - \frac{\|\mathbf{d}
        - \mathbf{d}_\mathbf{obs}\|^2}{\|\mathbf{d}_\mathbf{obs}\|^2}\right)
        \times 100\%\,,

    where :math:`\mathbf{d}_\mathbf{obs}` are the observed data, `dt`, and
    :math:`\mathbf{d}` are the synthetic data predicted by the forward model, `dtnew`.

    Args:
        dt: Array of original data
        dtnew: Array of modeled data

    Returns:
        float: Variance reduction :math:`\mathrm{VR}`
    """

    shp = np.shape(dt)
    shp = shp[0] * shp[1]
    dt_temp = np.reshape(dt, shp)
    dtnew_temp = np.reshape(dtnew, shp)
    d_dnew2 = (dt_temp - dtnew_temp) ** 2
    d2 = dt_temp**2
    VR = (1 - (np.sum(d_dnew2) / np.sum(d2))) * 100

    return VR


def _makeconvmat(c, size=None):
    r"""Build matrix used for convolution as implemented by matrix multiplication.

    Args:
        c (1D array): Signal to make convolution matrix for
        size (list or tuple): Optional input for desired size as ``(nrows, ncols)``;
            this will just shift ``cflip`` until it reaches the right size

    Returns:
        :class:`~numpy.ndarray`: Convolution matrix
    """

    cflip = c[::-1]  # Flip order
    if size is None:
        C = np.zeros((2 * len(c) - 1, 2 * len(c) - 1))
        for i in range(2 * len(c) - 1):
            if i > len(c) - 1:
                zros = np.zeros(i + 1 - len(c))
                p = np.concatenate((zros, cflip, np.zeros(2 * len(c))))
            else:
                p = np.concatenate(((cflip[-(i + 1) :]), np.zeros(2 * len(c))))
            p = p[: 2 * len(c) - 1]
            C[i, :] = p.copy()
    else:
        # Make it the correct size
        C = np.zeros(size)
        for i in range(size[0]):
            if i > len(c) - 1:
                zros = np.zeros(i + 1 - len(c))
                p = np.concatenate((zros, cflip, np.zeros(size[1])))
            else:
                p = np.concatenate(((cflip[-(i + 1) :]), np.zeros(size[1])))
            p = p[: size[1]]  # Cut p to the correct size
            C[i, :] = p.copy()

    return C


def _makeshiftmat(c, shiftby, size1):
    r"""Build matrix that can be used for shifting of overlapping triangles.

    Used for triangle method. Signal goes across rows and each shift is a new column
    (opposite orientation to :func:`_makeconvmat`)

    Args:
        c: Array of data (usually Green's function)
        shiftby (int): Number of samples to shift Green's function in each row
        size1 (list or tuple): Shape ``(nrows, ncols)`` of desired result. Will pad `c`
            if ``nrows`` is greater than ``len(c)``. Will shift `c` forward `shiftby`
            :math:`\times` ``ncols`` times

    Returns:
        :class:`~numpy.ndarray`: Matrix of shifted `c` of size `size1`
    """

    diff = len(c) - size1[0]
    if diff < 0:
        cpad = np.pad(c, (0, -diff), mode='edge')
    elif diff > 0:
        cpad = c[: size1[0]]
    else:
        cpad = c
    C = np.zeros(size1)
    for i in range(size1[1]):  # Loop over shifts and apply
        nshift = i * shiftby
        temp = np.pad(cpad.copy(), (nshift, 0), mode='edge')  # , end_values=(0., 0.))
        temp = temp[: size1[0]]
        C[:, i] = temp.copy()

    return C


def _curvature(x, y):
    r"""Estimate radius of curvature for each point on line to find corner of L-curve.

    Args:
        x: Array of x data
        y: Array of y data

    Returns:
        :class:`~numpy.ndarray`: Radius of curvature for each point (ends will be NaN)
    """

    # For each set of three points, find the radius of the circle that fits them (ignore
    # ends - these should be infinity since it's a straight line that fits them)
    R_2 = np.ones(len(x)) * float('inf')
    for i in range(1, len(R_2) - 1):
        xsub = x[i - 1 : i + 2]
        ysub = y[i - 1 : i + 2]
        # Slope of bisector of first segment
        m1 = -1 / ((ysub[0] - ysub[1]) / (xsub[0] - xsub[1]))
        # Slope of bisector of second segment
        m2 = -1 / ((ysub[1] - ysub[2]) / (xsub[1] - xsub[2]))
        # Compute b for first bisector
        b1 = ((ysub[0] + ysub[1]) / 2) - m1 * ((xsub[0] + xsub[1]) / 2)
        # Compute b for second bisector
        b2 = ((ysub[1] + ysub[2]) / 2) - m2 * ((xsub[1] + xsub[2]) / 2)

        Xc = (b1 - b2) / (m2 - m1)  # Find intercept point of bisectors
        Yc = b2 + m2 * Xc

        # Get distance from any point to intercept of bisectors to get radius
        R_2[i] = np.sqrt((xsub[0] - Xc) ** 2 + (ysub[0] - Yc) ** 2)
    return R_2


def _read(url):
    r"""Wrapper for :func:`obspy.core.stream.read` for long URLs."""
    with tempfile.NamedTemporaryFile() as f:
        urlretrieve(url, f.name)
        return read(f.name)


def _get_band_code(dt):
    r"""Determine SEED band code for a given sampling interval.

    SEED band code reference:
    https://www.fdsn.org/pdf/SEEDManual_V2.4_Appendix-A.pdf (see page 2)

    Code copied from Instaseis:
    https://github.com/krischer/instaseis/blob/dc9d4f16e55837236712e3dde2fbe10902393940/instaseis/helpers.py#L45-L61
    """
    if dt <= 0.001:
        band_code = 'F'
    elif dt <= 0.004:
        band_code = 'C'
    elif dt <= 0.0125:
        band_code = 'H'
    elif dt <= 0.1:
        band_code = 'B'
    elif dt < 1:
        band_code = 'M'
    else:
        band_code = 'L'
    return band_code


def readrun(filename):
    r"""Read in a saved LSForce object.

    Warning:
        Do not expect this to work if you have the ``autoreload``
        `IPython extension`_ enabled!

    Args:
        filename (str): File path to LSForce object saved using
            :meth:`~lsforce.lsforce.LSForce.saverun`

    Returns:
        :class:`~lsforce.lsforce.LSForce`: Saved LSForce object

    .. _IPython extension: https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
    """
    with open(filename, 'rb') as f:
        result = pickle.load(f)

    return result
