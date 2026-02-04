#!/home/jessica.irwin/.conda/envs/nds2utils/bin/python

# Copyright (C) 2017 Alex Nitz, Duncan Macleod
#               2022 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Generate a bank of templates using a brute force stochastic method.
EDITING: include tidal parameters
"""
import numpy as np
import logging
import argparse
import numpy.random
from scipy.stats import gaussian_kde

import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft, pycbc.conversions
import pycbc.pool
from pycbc import transforms
from pycbc.waveform.spa_tmplt import spa_length_in_time
from pycbc.distributions import read_params_from_config
from pycbc.distributions.utils import draw_samples_from_config, prior_from_config
from pycbc.io import HFile
import bilby

parser = argparse.ArgumentParser(description=__doc__)
pycbc.add_common_pycbc_options(parser)
parser.add_argument(
    "--output-file", required=True, help="Output file name for template bank."
)
parser.add_argument(
    "--input-file", help="Bank to use as an initial set of starting samples."
)
parser.add_argument(
    "--keep-entire-input-file",
    help="All of the templates in the input file will be kept in the output file, even if they do not pass normal mismatch criteria.",
    action="store_true",
)
parser.add_argument(
    "--input-config", help="Draw parameters from the given configure file."
)
parser.add_argument("--params", help="list of paramaters to use", nargs="+")
parser.add_argument(
    "--min", help="list of the minimum parameter values", nargs="+", type=float
)
parser.add_argument(
    "--max", help="list of the maximum parameter values", nargs="+", type=float
)
parser.add_argument(
    "--approximant", required=False, help="The waveform approximant to place."
)
parser.add_argument("--minimal-match", default=0.97, type=float)
parser.add_argument(
    "--buffer-length", default=2, type=float, help="size of waveform buffer in seconds"
)
parser.add_argument(
    "--full-resolution-buffer-length",
    default=None,
    type=float,
    help="Size of the waveform buffer in seconds for generating time-domain signals at full resolution before conversion to the frequency domain.",
)
parser.add_argument(
    "--max-signal-length",
    type=float,
    help="When specified, it cuts the maximum length of the waveform model to the lengh provided",
)
parser.add_argument(
    "--sample-rate", default=2048, type=float, help="sample rate in seconds"
)
parser.add_argument("--low-frequency-cutoff", default=20.0, type=float)
parser.add_argument("--enable-sigma-bound", action="store_true")
parser.add_argument("--tau0-threshold", type=float)
parser.add_argument(
    "--permissive", action="store_true", help="Allow waveform generator to fail."
)
parser.add_argument(
    "--placement-iterations",
    default=1000,
    type=int,
    help="Specify the number of attempts the bank should make when placing points. Use this option if the bank fails to place any points.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tolerance", type=float)
parser.add_argument("--max-mtotal", type=float)
parser.add_argument("--min-mchirp", type=float)
parser.add_argument("--max-mchirp", type=float)
parser.add_argument("--fixed-params", type=str, nargs="*")
parser.add_argument("--fixed-values", type=float, nargs="*")
parser.add_argument("--use-cross", action="store_true")
parser.add_argument("--max-q", type=float)
parser.add_argument("--tau0-crawl", type=float)
parser.add_argument("--tau0-start", type=float)
parser.add_argument("--tau0-end", type=float)
parser.add_argument("--tau0-cutoff-frequency", type=float, default=15.0)
parser.add_argument(
    "--nprocesses",
    type=int,
    default=1,
    help="Number of processes to use for waveform generation parallelization. If not given then only a single core will be used.",
)
pycbc.psd.insert_psd_option_group(parser)
parser.add_argument(
    "--use-trimmed-buffer",
    action="store_true",
    help=(
        "When specified, the match calculation will use only the first and last 100 samples "
        "of the waveform instead of the full buffer. Enabling this option makes match computation faster."
    ),
)


args = parser.parse_args()

print("local version")

pycbc.init_logging(args.verbose)
numpy.random.seed(args.seed)

# Read the .ini file if it's in the input.
if args.input_config is not None:
    config_parser = pycbc.types.config.InterpolatingConfigParser()
    file = open(args.input_config, "r")
    config_parser.read_file(file)
    file.close()

    variable_args, static_args = read_params_from_config(
        config_parser,
        prior_section="prior",
        vargs_section="variable_params",
        sargs_section="static_params",
    )

    if any(config_parser.get_subsections("waveform_transforms")):
        waveform_transforms = transforms.read_transforms_from_config(
            config_parser, "waveform_transforms"
        )
    else:
        waveform_transforms = None

    dists_joint = prior_from_config(cp=config_parser)

fdict = {}
if args.fixed_params:
    fdict = {p: v for (p, v) in zip(args.fixed_params, args.fixed_values)}


class Shrinker(object):
    def __init__(self, data):
        self.data = data

    def pop(self):
        if len(self.data) == 0:
            return None
        l = self.data[-1]
        self.data = self.data[:-1]
        return l
        
class TriangleBank(object):
    """ A bank of templates that uses the triangle inequality to estimate
    matches based on prior ones.
    """
    def __init__(self, p=None):
        self.waveforms = p if p is not None else []
        self.tbins = {}

    def __len__(self):
        return len(self.waveforms)

    def activelen(self):
        i = 0
        for w in self.waveforms:
            if isinstance(w, pycbc.types.FrequencySeries):
                i += 1
        return i

    def insert(self, hp):
        self.waveforms.append(hp)

        for b in [hp.tbin - 1, hp.tbin, hp.tbin + 1]:
            if b in self.tbins:
                self.tbins[b].append(len(self)-1)
            else:
                self.tbins[b] = [len(self)-1]

    def __getitem__(self, index):
        return self.waveforms[index]

    def keys(self):
        # print('in class')
        # print(self.waveforms[0].params.keys())
        # print('params')
        # print(params.keys())
        return list(params.keys()) #self.waveforms[0].params.keys()
        
    def component_keys(self):
        # print('in class')
        # print(self.waveforms[0].params.keys())
        # print('params')
        # print(params.keys())
        return self.waveforms[0].params.keys()

    def component_key(self, k):
        return numpy.array([p.params[k] for p in self.waveforms])

    def key(self, k):
        # # p is a saved waveform
        # # M is total mass 
        # # so gets the data associated to a single key
        # for p in self.waveforms:
        #     print(p)
        #     print(np.shape(self.wav
        #     print(k)
        #     print(numpy.array(p.params[k]))
        #     exit(0)
        return numpy.array(params[k])

    def sigma_match_bound(self, sig):
        if not hasattr(self, 'sigma'):
            self.sigma = None
        if self.sigma is None or len(self.sigma) != len(self):
            self.sigma = numpy.array([h.s for h in bank.waveforms])
        return numpy.minimum(sig / self.sigma, self.sigma / sig)

    def range(self):
        if not hasattr(self, 'r'):
            self.r = None
        if self.r is None or len(self.r) != len(self):
            self.r = numpy.arange(0, len(self))
        return self.r

    def culltau0(self, threshold):
        cull = numpy.where(self.tau0() < threshold)[0]

        class dumb(object):
            pass
        for c in cull:
            d = dumb()
            d.tau0 = self.waveforms[c].tau0
            d.params = self.waveforms[c].params
            d.s = self.waveforms[c].s
            self.waveforms[c] = d

    def tau0(self):
        if not hasattr(self, 't0'):
            self.t0 = None
        if self.t0 is None or len(self.t0) != len(self):
            self.t0 = numpy.array([h.tau0 for h in self])
        return self.t0

    def __contains__(self, hp):
        mmax = 0
        mnum = 0
        #Apply sigmas maximal match.
        if args.enable_sigma_bound:
            matches = self.sigma_match_bound(hp.s)
            r = self.range()[matches > hp.threshold]
        else:
            matches = numpy.ones(len(self))
            r = self.range()

        msig = len(r)

        #Apply tau0 threshold
        if args.tau0_threshold:
            hp.tau0 = pycbc.conversions.tau0_from_mass1_mass2(
                                            hp.params['mass1'],
                                            hp.params['mass2'],
                                            args.tau0_cutoff_frequency)
            hp.tbin = int(hp.tau0 / args.tau0_threshold)

            if hp.tbin in self.tbins:
                r = numpy.array(self.tbins[hp.tbin])
            else:
                r = r[:0]

        mtau = len(r)

        # Try to do some actual matches
        inc = Shrinker(r*1)
        while 1:
            j = inc.pop()
            if j is None:
                hp.matches = matches[r]
                hp.indices = r
                logging.info("TADD MaxMatch:%0.3f Size:%i "
                             "AfterSigma:%i AfterTau0:%i Matches:%i"
                              % (mmax, len(self), msig, mtau, mnum))
                return False

            hc = self[j]

            # Defensive initialization if matches/indices are missing
            if not hasattr(hc, 'matches'):
                hc.matches = numpy.empty(len(self))
                hc.matches[:] = numpy.nan

            if not hasattr(hc, 'indices'):
                hc.indices = numpy.arange(len(self))


            m = hp.gen.match(hp, hc)
            matches[j] = m
            mnum += 1

            # Update bounding match values, apply triangle inequality
            maxmatches = hc.matches - m + 1.10
            update = numpy.where(maxmatches < matches[hc.indices])[0]
            matches[hc.indices[update]] = maxmatches[update]

            # Update where to calculate matches
            skip_threshold = 1 - (1 - hp.threshold) * 2.0
            inc.data = inc.data[matches[inc.data] > skip_threshold]

            if m > hp.threshold:
                return True
            if m > mmax:
                mmax = m

    def check_params(self, gen, params, threshold, force_add=False):
        num_added = 0
        total_num = len(tuple(params.values())[0])
        waveform_cache = []

        pool = pycbc.pool.choose_pool(args.nprocesses)
        for return_wf in pool.imap_unordered(
                wf_wrapper,
                ({k: params[k][idx] for k in params} for idx in range(total_num))
            ):
            waveform_cache += [return_wf]
        pool.close_pool()
        del pool

        idx_tracker = 0
        indices_kept = np.array([])

        for hp in waveform_cache:
            if hp is not None:
                hp.gen = gen
                hp.threshold = threshold
                if hp not in self:
                    num_added += 1
                    indices_kept = np.append(indices_kept, idx_tracker)
                    self.insert(hp)
                elif force_add:
                    num_added += 1
                    self.insert(hp)
            else:
                logging.info("Waveform generation failed!")
                continue
            idx_tracker += 1
            
        return bank, num_added / total_num, indices_kept.astype(int)

def decimate_frequency_domain(template, target_df):
    """
    Returns a frequency-domain waveform resampled to a lower frequency resolution
    (delta_f) by decimation.

    Parameters
    ----------
    template : pycbc.types.FrequencySeries
        The input frequency-domain signal to be decimated.
    target_df : float
        The target frequency resolution (delta_f) for the decimated signal.

    Returns
    ----------
    decimated_template : pycbc.types.FrequencySeries
        A new FrequencySeries object with the decimated data and the specified
        target delta_f.
    """
    # Calculate the decimation factor
    decimation_factor = int(target_df / template.delta_f)

    if decimation_factor < 1:
        raise ValueError(
            "Target delta_f must be greater than or equal to the original delta_f."
        )

    # Decimate the data by selecting every 'decimation_factor'-th point
    decimated_signal = template.data[::decimation_factor]

    # Create a new FrequencySeries object with the decimated data and the target delta_f
    decimated_template = pycbc.types.FrequencySeries(
        decimated_signal, delta_f=target_df
    )
    return decimated_template


class GenUniformWaveform(object):
    def __init__(self, buffer_length, sample_rate, f_lower):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate)
        self.flen = tlen // 2 + 1
        psd = pycbc.psd.from_cli(args, self.flen, self.delta_f, self.f_lower)
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin : -1]) ** 0.5).astype(numpy.float32)
        qtilde = pycbc.types.zeros(tlen, numpy.complex64)
        q = pycbc.types.zeros(tlen, numpy.complex64)
        self.qtilde_view = qtilde[self.kmin : self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)

        self.md = q
        self.md2 = numpy.zeros(20)

        if args.use_trimmed_buffer:
            self.md = q._data[-100:]
            self.md2 = q._data[0:100]

    def generate(self, **kwds):
        kwds.update(fdict)
        if args.max_signal_length is not None:
            flow = numpy.arange(self.f_lower, 100, 0.1)[::-1]
            length = spa_length_in_time(
                mass1=kwds["mass1"], mass2=kwds["mass2"], f_lower=flow, phase_order=-1
            )
            maxlen = args.max_signal_length
            x = numpy.searchsorted(length, maxlen) - 1
            l = length[x]
            f = flow[x]
        else:
            f = self.f_lower

        kwds["f_lower"] = f
        if hasattr(kwds["approximant"], "decode"):
            kwds["approximant"] = kwds["approximant"].decode()

        if kwds["approximant"] in pycbc.waveform.fd_approximants():
            if args.full_resolution_buffer_length is not None:
                # Generate the frequency-domain waveform at full frequency resolution
                high_hp, high_hc = pycbc.waveform.get_fd_waveform(
                    delta_f=1 / args.full_resolution_buffer_length, **kwds
                )
                # Decimate the generated signal to a reduced frequency resolution
                hp = decimate_frequency_domain(high_hp, 1 / args.buffer_length)
                hc = decimate_frequency_domain(high_hc, 1 / args.buffer_length)
            else:
                hp, hc = pycbc.waveform.get_fd_waveform(delta_f=self.delta_f, **kwds)
            if args.use_cross:
                hp = hc

            if "fratio" in kwds:
                hp = hc * kwds["fratio"] + hp * (1 - kwds["fratio"])

        else:
            dt = 1.0 / args.sample_rate
            hp = pycbc.waveform.get_waveform_filter(
                pycbc.types.zeros(self.flen, dtype=numpy.complex64),
                delta_f=self.delta_f,
                delta_t=dt,
                **kwds
            )

        hp.resize(self.flen)
        hp = hp.astype(numpy.complex64)
        hp[self.kmin : -1] *= self.w
        s = float(1.0 / pycbc.filter.sigmasq(hp, low_frequency_cutoff=f) ** 0.5)
        hp *= s
        hp.params = kwds
        hp.view = hp[self.kmin : -1]
        hp.s = (1.0 / s) ** 2.0
        
        return hp

    def match(self, hp, hc):
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(self.md).max(), abs(self.md2).max())
        return m * 4.0 * self.delta_f


r = 0
if not args.tolerance:
    tolerance = (1 - args.minimal_match) / 10
else:
    tolerance = args.tolerance

size = int(1.0 / tolerance)

gen = GenUniformWaveform(
    args.buffer_length, args.sample_rate, args.low_frequency_cutoff
)

# initialise bank
bank = TriangleBank()

def wf_wrapper(p):
    try:
        hp = gen.generate(**p)
        return hp
    except Exception as e:
        print(e)
        return None


if args.input_file:
    # not true for now
    f = HFile(args.input_file, "r")
    params = {k: f[k][:] for k in f}
    bank, _, idxs = bank.check_params(
        gen, params, args.minimal_match, force_add=args.keep_entire_input_file
    )


def get_mass_components(M, q):
    m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(q, M)
    return m1, m2


def get_tidal_components(lambdatilde, mass1, mass2):
    l1, l2 = bilby.gw.conversion.lambda_tilde_to_lambda_1_lambda_2(
        lambdatilde, mass1, mass2
    )
    return l1, l2


def get_waveform_params_dict(params):
    m1, m2 = get_mass_components(params["M"], params["q"])
    l1, l2 = get_tidal_components(params["lambdatilde"], m1, m2)
    # adding approximant too!!!!
    return {
        "mass1": m1,
        "mass2": m2,
        "lambda1": l1,
        "lambda2": l2,
        "approximant": params["approximant"],
    }

def get_waveform_params(params):
    m1, m2 = get_mass_components(params[0], params[1])
    l1, l2 = get_tidal_components(params[2], m1, m2)
    # adding approximant too!!!!
    return np.array(m1,m2,l1,l2)

def get_bank_params(params):
    M = bilby.gw.conversion.component_masses_to_total_mass(params[0], params[1])
    q = bilby.gw.conversion.component_masses_to_mass_ratio(params[0], params[1])
    lambdatilde = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(params[2], params[3], params[0], params[1])
    # adding approximant too!!!!
    return np.array([M,q,lambdatilde])

def draw(rtype, reparams):

    if rtype == "uniform":
        if args.input_config is None:

            # still want to draw bank in our original parameterisation
            # args.params are just M q lambda tilde
            params = {
                name: numpy.random.uniform(pmin, pmax, size=size)
                for name, pmin, pmax in zip(args.params, args.min, args.max)
            }
        else:
            # `draw_samples_from_config` has its own fixed seed, so must overwrite it.
            random_seed = numpy.random.randint(low=0, high=2**32 - 1)
            samples = draw_samples_from_config(args.input_config, size, random_seed)
            params = {name: samples[name] for name in samples.fieldnames}
            # params also the three we want
            # Add `static_args` back.
            if static_args is not None:
                for k in static_args.keys():
                    params[k] = numpy.array([static_args[k]] * size)

    elif rtype == "kde":

        trail = 300
        if trail > len(bank):
            trail = len(bank)
        p_reparam = list(reparams.keys())
        p = bank.component_keys()
        # these keys are components
        p = [k for k in p if k not in fdict]
        p.remove("approximant")
        p_reparam.remove("f_lower")
        if args.input_config is not None:
            # not relevant
            p = variable_args
        # bdata = numpy.array([reparams.key(k)[-trail:] for k in p])
        # for k in p:
        #     bdata = np.array(bank.component_key(k)[-trail:])
        bdata = numpy.array([bank.component_key(k)[-trail:] for k in p])
        bdata_reparam = get_bank_params(bdata)
        kde = gaussian_kde(bdata_reparam)
        points = kde.resample(size=size)
        params = {k: v for k, v in zip(p_reparam, points)}

        # Add `static_args` back, some transformations may need them.
        if args.input_config is not None and static_args is not None:
            for k in static_args.keys():
                params[k] = numpy.array([static_args[k]] * size)

        # Apply `waveform_transforms` defined in the .ini file to samples.
        # not applicable
        if args.input_config is not None and waveform_transforms is not None:
            params = transforms.apply_transforms(params, waveform_transforms)

    # no issue here
    if args.approximant is not None:
        params["approximant"] = numpy.array([args.approximant] * size)

    # Filter out stuff (kde method may also generate samples outside boundaries).
    l = None
    if args.input_config is None:
        # this is true for us rn
        # keep everything in boundaries

        for name, pmin, pmax in zip(args.params, args.min, args.max):
            # key error here idk why
            nl = (params[name] < pmax) & (params[name] > pmin)
            l = (nl & l) if l is not None else nl

        # if args.max_q:
        #     q = numpy.maximum(
        #         params["mass1"] / params["mass2"], params["mass2"] / params["mass1"]
        #     )
        #     l &= q < args.max_q

        # if args.max_mtotal:
        #     l &= params["mass1"] + params["mass2"] < args.max_mtotal

        # if args.max_mchirp:
        #     from pycbc.conversions import mchirp_from_mass1_mass2

        #     mc = mchirp_from_mass1_mass2(params["mass1"], params["mass2"])
        #     l &= mc < args.max_mchirp

        # if args.min_mchirp:
        #     from pycbc.conversions import mchirp_from_mass1_mass2

        #     mc = mchirp_from_mass1_mass2(params["mass1"], params["mass2"])
        #     l &= mc > args.min_mchirp

    else:
        # first call of component mass parameters
        l = dists_joint.contains(params)

    params = {k: params[k][l] for k in params}
    # M, q, lambda tilde, approximant.

    # len is 200 here always
    
    return params


def cdraw(rtype, ts, te, reparams):

    # rtype is uniform

    from pycbc.conversions import tau0_from_mass1_mass2

    # returns dictionary of  M, q, lambda tilde, approximant.
    # draws points in the bank in the M, q, lambda tilde space
    p = draw(rtype,reparams)

    # getting component masses to calculate taus
    if "M" in p and "q" in p:
        # if mass is not in components, get components
        mass1, mass2 = get_mass_components(p["M"], p["q"])
    else:
        mass1 = p["mass1"]
        mass2 = p["mass2"]
        
    # len(p[list(p.keys())[0]]) len 200
    # size also 200

    # calculate taus and start filling 
    if len(p[list(p.keys())[0]]) > 0:
        t = tau0_from_mass1_mass2(mass1, mass2, args.tau0_cutoff_frequency)
        # boolean array
        l = (t < te) & (t > ts)
        for k in p:
            p[k] = p[k][l]

    i = 0
    while len(p[list(p.keys())[0]]) < size:

        tp = draw(rtype, reparams)
        for k in p:
            # loops through the different parameters
            # i.e. starts on total mass
            # adds total mass drawn from tp to p
            p[k] = numpy.concatenate([p[k], tp[k]])

        # need to get component masses from the updated p
        if "M" in p and "q" in p:
            # if mass is not in components, get components
            mass1, mass2 = get_mass_components(p["M"], p["q"])
        else:
            mass1 = p["mass1"]
            mass2 = p["mass2"]

        if len(p[list(p.keys())[0]]) > 0:
            t = tau0_from_mass1_mass2(mass1, mass2, args.tau0_cutoff_frequency)
            # think the error was here, the boolean array is l, which is calculated given t
            # which is calculated given mass1 and mass2, which we hadn't recalculated since the start
            # of the function. Need to recalculate given p has been updated with tp, and then run 
            # this similar piece of code.
            l = (t < te) & (t > ts)
            #p = {k: p[k][l] for k in p}
            for k in p:
                p[k] = p[k][l]

        i += 1
        if i > args.placement_iterations:
            break

    if len(p[list(p.keys())[0]]) == 0:
        return None

    # p is params
    return p


tau0s = args.tau0_start
tau0e = tau0s + args.tau0_crawl

go = True
params = {}

bank_params = {'M':[], 'q':[], 'lambdatilde':[]}

region = 0

import time
start_time = time.time()
while tau0s < args.tau0_end:
    conv = 1
    r = 0
    while conv > tolerance:
        # Standard Round
        r += 1
        params = cdraw("uniform", tau0s, tau0e, params)
        if params is None:
            if len(bank) > 0:
                go = False
            break
        
        blen = len(bank)
        # len bank is 0
        # change bank keys
        # params are M, q, lambdatilde
        # new params
        params['f_lower'] = args.low_frequency_cutoff
        component_params = get_waveform_params_dict(params)

        # build bank given components so that we check waveform
        bank, uconv, keep_idxs = bank.check_params(gen, component_params, args.minimal_match)

        for key in ('M', 'q', 'lambdatilde'):
            if len(bank_params['M']) == 0:
                bank_params[key] = np.take(np.array(params[key]), keep_idxs)
            else:
            # update bank params with more indices of the next set of params
                bank_params[key] = np.append(bank_params[key], np.take(np.array(params[key]), keep_idxs))
        
        # bank still object
        logging.info(
            "%s: Round (U): %s Size: %s conv: %s added: %s",
            region,
            r,
            len(bank),
            uconv,
            len(bank) - blen,
        )
        if r > 10:
            conv = uconv
        kloop = 0

        while ((kloop == 0) or (kconv / okconv) > 0.5) and len(bank) > 10:
            r += 1
            kloop += 1
            # checking this function

            # GO FROM HERE
            # bank keys should be M, q, lambdatilde not
            # m1 m2 l1 l2
            params['f_lower'] = args.low_frequency_cutoff
            #params = cdraw("kde", tau0s, tau0e, params)
            new_params = cdraw("uniform", tau0s, tau0e, params)

            if new_params is not None:
                params = new_params
            else:
                # Handle the case where no parameters were drawn
                if len(bank) > 0:
                    go = False
                break
                
            blen = len(bank)

            component_params = get_waveform_params_dict(params)
            bank, kconv, keep_idxs = bank.check_params(gen, component_params, args.minimal_match)

            for key in ('M', 'q', 'lambdatilde'):
                if len(bank_params[key]) == 0:
                    # print('bank params empty')
                    bank_params[key] = np.take(np.array(params[key]), keep_idxs)
                else:
                    # update bank params with more indices of the next set of params
                    # print('bank params exists')
                    bank_params[key] = np.append(bank_params[key], np.take(np.array(params[key]), keep_idxs))

            # need to get bank.key to work!!!
            # bank convert
            
            logging.info(
                "%s: Round (K) (%s): %s Size: %s conv: %s added: %s",
                region,
                kloop,
                r,
                len(bank),
                kconv,
                len(bank) - blen,
            )

            if uconv:
                logging.info("Ratio of convergences: %2.3f" % (kconv / (uconv)))
                logging.info("Progress: {:.0%} completed".format(tau0e / args.tau0_end))
                print('time taken', time.time() - start_time)

            if kloop == 1:
                okconv = kconv

            if kconv <= tolerance:
                conv = kconv
                break         

    bank.culltau0(tau0s - args.tau0_threshold * 2.0)
    logging.info("Region Done %3.1f-%3.1f, %s stored", tau0s, tau0e, bank.activelen())
    region += 1
    tau0s += args.tau0_crawl / 2
    tau0e += args.tau0_crawl / 2

    
# o = HFile(args.output_file, "w")
# o.attrs["minimal_match"] = args.minimal_match
# for k in keys:

#     # the code will get to here and will stop working
#     # so need to fix this
    
#     val = bank.key(k)
#     print('number of templates', len(val))
#     if val.dtype.char == "U":
#         val = val.astype("bytes")
#     o[k] = val
# o.close()

o = HFile(args.output_file, "w")
o.attrs["minimal_match"] = args.minimal_match
for k in list(bank_params.keys()):

    # the code will get to here and will stop working
    # so need to fix this
    
    val = bank_params[k]
    print('number of templates', len(val))
    if val.dtype.char == "U":
        val = val.astype("bytes")
    o[k] = val
o.close()

print('end of bank creation')
exit(0)

