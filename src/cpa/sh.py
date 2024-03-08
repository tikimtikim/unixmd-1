from __future__ import division
from build.el_propagator import el_run
from cpa.cpa import CPA
from misc import eps, au_to_K, call_name, typewriter
import random, os, shutil, textwrap
import numpy as np
import pickle

class SH(CPA):
    """ Class for surface hopping dynamics with CPA

        :param object molecule: Molecule object
        :param integer istate: Initial state
        :param double dt: Time interval
        :param integer nsteps: Total step of nuclear propagation
        :param integer nesteps: Total step of electronic propagation
        :param string elec_object: Electronic equation of motions
        :param string propagator: Electronic propagator
        :param boolean l_print_dm: Logical to print BO population and coherence
        :param boolean l_adj_nac: Adjust nonadiabatic coupling to align the phases
        :param init_coef: Initial BO coefficient
        :type init_coef: double, list or complex, list
        :param string dec_correction: Simple decoherence correction schemes
        :param double edc_parameter: Energy constant (H) for rescaling coefficients in edc
        :param string unit_dt: Unit of time step 
        :param integer out_freq: Frequency of printing output
        :param integer verbosity: Verbosity of output
    """
    def __init__(self, molecule, istate=0, dt=0.5, nsteps=1000, nesteps=20, \
        elec_object="density", propagator="rk4", l_print_dm=True, l_adj_nac=True, init_coef=None, \
        dec_correction=None, edc_parameter=0.1, unit_dt="fs", out_freq=1, verbosity=0):
        # Initialize input values
        super().__init__(molecule, istate, dt, nsteps, nesteps, \
            elec_object, propagator, l_print_dm, l_adj_nac, init_coef, unit_dt, out_freq, verbosity)

        # Initialize SH variables
        self.rstate = istate
        self.rstate_old = self.rstate

        self.rand = 0.
        self.prob = np.zeros(self.mol.nst)
        self.acc_prob = np.zeros(self.mol.nst + 1)

        self.l_hop = False
        self.l_reject = False

        # Initialize decoherence variables
        self.dec_correction = dec_correction
        self.edc_parameter = edc_parameter

        if (self.dec_correction != None):
            self.dec_correction = self.dec_correction.lower()

        if not (self.dec_correction in [None, "idc", "edc"]):
            error_message = "Invalid decoherence corrections in FSSH method!"
            error_vars = f"dec_correction = {self.dec_correction}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Debug variables
        self.dotpopnac = np.zeros(self.mol.nst)

        # Initialize event to print
        self.event = {"HOP": []}

    def run(self, qm, mm=None, output_dir="./", l_save_qm_log=False, l_save_mm_log=False, l_save_scr=True, restart=None):
        """ Run CPA dynamics according to surface hopping dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string output_dir: Name of directory where outputs to be saved.
            :param boolean l_save_qm_log: Logical for saving QM calculation log
            :param boolean l_save_mm_log: Logical for saving MM calculation log
            :param boolean l_save_scr: Logical for saving scratch directory
            :param string restart: Option for controlling dynamics restarting
        """
        # Initialize PyUNIxMD
        base_dir, unixmd_dir, qm_log_dir, mm_log_dir =\
             self.run_init(qm, mm, output_dir, l_save_qm_log, l_save_mm_log, l_save_scr, restart)

        bo_list = [] # a redundant variable in CPA-like dynamics
        qm.calc_coupling = False
        self.print_init(qm, mm, restart)

        if (restart == None):
            pass
        elif (restart == "write"):
            pass
        elif (restart == "append"):
            pass
        self.istep += 1
        
        # Main MD loop
        for istep in range(self.istep, self.nsteps):
            pass
           
    def hop_prob(self, istep):
        """ Routine to calculate hopping probabilities

            :param integer istep: Current MD step
        """
        pass

    def hop_check(self, bo_list):
        """ Routine to check hopping occurs with random number

            :param integer,list bo_list: List of BO states for BO calculation
        """
        pass

    def correct_dec_idc(self):
        """ Routine to decoherence correction, instantaneous decoherence correction(IDC) scheme
        """
        pass


    def correct_dec_edc(self):
        """ Routine to decoherence correction, energy-based decoherence correction(EDC) scheme
        """
        pass

    def update_energy(self):
        """ Routine to update the energy of molecules in surface hopping dynamics
        """
        pass

    def write_md_output(self, unixmd_dir, istep):
        """ Write output files

            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step

        """
        pass

    def write_sh(self, unixmd_dir, istep):
        """ Write hopping-related quantities into files

            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        pass

    def write_dotpop(self, unixmd_dir, istep):
        """ Write time-derivative BO population
           
            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        pass

    def print_init(self, qm, mm, restart):
        """ Routine to print the initial information of dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string restart: Option for controlling dynamics restarting
        """
        # Print initial information about molecule, qm, mm and thermostat
        super().print_init(qm, mm, restart)

        # Print dynamics information for start line
        dynamics_step_info = textwrap.dedent(f"""\

        {"-" * 118}
        {"Start Dynamics":>65s}
        {"-" * 118}
        """)

        # Print INIT for each step
        INIT = f" #INFO{'STEP':>8s}{'State':>7s}{'Kinetic(H)':>14s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}{'Norm.':>8s}"
        dynamics_step_info += INIT

        # Print DEBUG1 for each step
        if (self.verbosity >= 1):
            DEBUG1 = f" #DEBUG1{'STEP':>6s}{'Rand.':>11s}{'Acc. Hopping Prob.':>28s}"
            dynamics_step_info += "\n" + DEBUG1

        print (dynamics_step_info, flush=True)

    def print_step(self, istep):
        """ Routine to print each steps infomation about dynamics

            :param integer istep: Current MD step
        """
        pass

    def read_QM_from_file(self, istep):
        """Routine to read precomputed QM information for CPA dynamics

           :param integer istep: Current MD step
        """
        pass

    def read_RP_from_file(self, istep):
        """Routine to read precomputed atomic position, velocities for CPA dynamics

           :param integer istep: Current MD step
        """
        pass
