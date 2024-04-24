from __future__ import division
from misc import fs_to_au, au_to_A, call_name, typewriter
import textwrap, datetime
import numpy as np
import os, shutil


class CPA(object):
    """ Class for electronic propagator used in MQC dynamics with classical path approximation

        :param object molecule: Molecule object
        :param integer istate: Initial adiabatic state
        :param double dt: Time interval
        :param integer nsteps: Nuclear step
        :param integer nesteps: Electronic step
        :param string elec_object: Electronic equation of motions
        :param string propagator: Electronic propagator
        :param boolean l_print_dm: Logical to print BO population and coherence
        :param boolean l_adj_nac: Logical to adjust nonadiabatic coupling
        :param init_coef: Initial BO coefficient
        :type init_coef: Double, list or complex, list
        :param string unit_dt: Unit of time step (fs = femtosecond, au = atomic unit)
        :param integer out_freq: Frequency of printing output
        :param integer verbosity: Verbosity of output
    """
    def __init__(self, molecule, istate, dt, nsteps, nesteps, \
        elec_object, propagator, l_print_dm, l_adj_nac, init_coef, unit_dt, out_freq, verbosity):
        # Save name of MQC dynamics
        self.md_type = self.__class__.__name__

        # Initialize Molecule object
        self.mol = molecule

        # Initialize input values
        self.istate = istate
        self.nsteps = nsteps
        self.nesteps = nesteps

        # Initialize time step
        self.istep = -1
        self.fstep = -1

        # Decide unit of time step
        self.unit_dt = unit_dt.lower()
        if (self.unit_dt == 'au'):
            self.dt = dt
        elif (self.unit_dt == 'fs'):
            self.dt = dt * fs_to_au
        else:
            error_message = "Invalid unit for time step!"
            error_vars = f"unit_dt = {unit_dt}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        self.elec_object = elec_object
        if (self.elec_object != None):
            self.elec_object = self.elec_object.lower()

        if not (self.elec_object in [None, "coefficient", "density"]):
            error_message = "Invalid electronic object!"
            error_vars = f"elec_object = {self.elec_object}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        self.propagator = propagator
        if (self.propagator != None):
            self.propagator = self.propagator.lower()

        if not (self.propagator in [None, "rk4", "exponential"]):
            error_message = "Invalid electronic propagator!"
            error_vars = f"propagator = {self.propagator}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        if (self.propagator == "exponential" and self.elec_object != "coefficient"):
            error_message = "exponential propagator is incompatible with objects other than coefficient"
            error_vars = f"elec_object = {self.elec_object}, propagator = {self.propagator}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        self.l_print_dm = l_print_dm

        self.l_adj_nac = l_adj_nac

        self.out_freq = out_freq
        self.verbosity = verbosity

        # Initialize coefficients and densities
        self.mol.get_coefficient(init_coef, self.istate)

    def run_init(self, qm, mm, output_dir, l_save_qm_log, l_save_mm_log, l_save_scr, restart):
        """ Initialize MQC dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string output_dir: Location of input directory
            :param boolean l_save_qm_log: Logical for saving QM calculation log
            :param boolean l_save_mm_log: Logical for saving MM calculation log
            :param boolean l_save_scr: Logical for saving scratch directory
            :param string restart: Option for controlling dynamics restarting
        """
        # Check whether the restart option is right
        if (restart != None):
            restart = restart.lower()

        if not (restart in [None, "write", "append"]):
            error_message = "Invalid restart option!"
            error_vars = f"restart = {restart}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Check if NACVs are calculated for Ehrenfest dynamics
        if (self.md_type == "Eh" and self.mol.l_nacme):
            error_message = "Ehrenfest dynamics needs evaluation of NACVs, check your QM object!"
            error_vars = f"(QM) qm_prog.qm_method = {qm.qm_prog}.{qm.qm_method}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Check compatibility of variables for QM and MM calculation
        if ((self.mol.l_qmmm and mm == None) or (not self.mol.l_qmmm and mm != None)):
            error_message = "Both logical for QM/MM and MM object is necessary!"
            error_vars = f"Molecule.l_qmmm = {self.mol.l_qmmm}, mm = {mm}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
        if (self.mol.l_qmmm and mm != None):
            self.check_qmmm(qm, mm)

        # Exception for CTMQC/Ehrenfest with QM/MM
        if ((self.md_type in ["CT", "Eh"]) and (mm != None)):
            error_message = "QM/MM calculation is not compatible with CTMQC or Ehrenfest now!"
            error_vars = f"mm = {mm}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Set directory information
        output_dir = os.path.expanduser(output_dir)
        base_dir = []
        unixmd_dir = []
        qm_log_dir = []
        mm_log_dir = [None]

        if (self.mol.l_qmmm and mm != None):
            mm_log_dir = []

        dir_tmp = os.path.join(os.getcwd(), output_dir)
        base_dir.append(dir_tmp)

        for idir in base_dir:
            unixmd_dir.append(os.path.join(idir, "md"))
            qm_log_dir.append(os.path.join(idir, "qm_log"))
            if (self.mol.l_qmmm and mm != None):
                mm_log_dir.append(os.path.join(idir, "mm_log"))

        # Check and make directories
        if (restart == "append"):
            # For MD output directory
            for md_idir in unixmd_dir:
                if (not os.path.exists(md_idir)):
                    error_message = f"Directory {md_idir} to be appended for restart not found!"
                    error_vars = f"restart = {restart}, output_dir = {output_dir}"
                    raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

            # For QM output directory
            if (l_save_qm_log):
                for qm_idir in qm_log_dir:
                    if (not os.path.exists(qm_idir)):
                        os.makedirs(qm_idir)

            # For MM output directory
            if (self.mol.l_qmmm and mm != None):
                if (l_save_mm_log):
                    for mm_idir in mm_log_dir:
                        if (not os.path.exists(mm_idir)):
                            os.makedirs(mm_idir)
        else:
            # For MD output directory
            for md_idir in unixmd_dir:
                if (os.path.exists(md_idir)):
                    shutil.move(md_idir, md_idir + "_old_" + str(os.getpid()))
                os.makedirs(md_idir)

                self.touch_file(md_idir)

            # For QM output directory
            for qm_idir in qm_log_dir:
                if (os.path.exists(qm_idir)):
                    shutil.move(qm_idir, qm_idir + "_old_" + str(os.getpid()))
                if (l_save_qm_log):
                    os.makedirs(qm_idir)

            # For MM output directory
            for mm_idir in mm_log_dir:
                if (self.mol.l_qmmm and mm != None):
                    if (os.path.exists(mm_idir)):
                        shutil.move(mm_idir, mm_idir + "_old_" + str(os.getpid()))
                    if (l_save_mm_log):
                        os.makedirs(mm_idir)

        os.chdir(base_dir[0])

        return base_dir[0], unixmd_dir[0], qm_log_dir[0], mm_log_dir[0]

    def cl_update_position(self):
        """ Routine to update nuclear positions
        """
        self.mol.vel += 0.5 * self.dt * self.rforce / np.column_stack([self.mol.mass] * self.mol.ndim)
        self.mol.pos += self.dt * self.mol.vel

    def cl_update_velocity(self):
        """ Routine to update nuclear velocities
        """
        self.mol.vel += 0.5 * self.dt * self.rforce / np.column_stack([self.mol.mass] * self.mol.ndim)
        self.mol.update_kinetic()

    def update_potential(self):
        """ Routine to update the potential of molecules
        """
        pass

    def print_init(self, qm, mm, restart):
        """ Routine to print the initial information of dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string restart: Option for controlling dynamics restarting
        """
        # Print PyUNIxMD version
        cur_time = datetime.datetime.now()
        cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
        prog_info = textwrap.dedent(f"""\
        {"-" * 68}

        {"PyUNIxMD version 20.1":>43s}

        {"< Developers >":>40s}
        {" " * 4}Seung Kyu Min,  In Seong Lee,  Jong-Kwon Ha,  Daeho Han,
        {" " * 4}Kicheol Kim,  Tae In Kim,  Sung Wook Moon

        {"-" * 68}

        {" " * 4}Please cite PyUNIxMD as follows:
        {" " * 4}I. S. Lee, J.-K. Ha, D. Han, T. I. Kim, S. W. Moon, & S. K. Min.
        {" " * 4}PyUNIxMD: A Python-based excited state molecular dynamics package.
        {" " * 4}Journal of Computational Chemistry, 42:1755-1766. 2021

        {" " * 4}PyUNIxMD begins on {cur_time}
        """)
        print (prog_info, flush=True)

        # Print restart info
        if (restart != None):
            restart_info = textwrap.indent(textwrap.dedent(f"""\
            Dynamics is restarted from the last step of a previous dynamics.
            Restart Mode: {restart}
            """), "    ")
            print (restart_info, flush=True)

        # Print self.mol information: coordinate, velocity
        self.mol.print_init(mm)

        # Print dynamics information
        dynamics_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"Dynamics Information":>43s}
        {"-" * 68}
          QM Program               = {qm.qm_prog:>16s}
          QM Method                = {qm.qm_method:>16s}
        """)
        if (self.mol.l_qmmm and mm != None):
            dynamics_info += textwrap.indent(textwrap.dedent(f"""\
              MM Program               = {mm.mm_prog:>16s}
              QMMM Scheme              = {mm.scheme:>16s}
            """), "  ")
            # Print charge embedding in MM program
            if (mm.embedding != None):
                dynamics_info += f"  Charge Embedding         = {mm.embedding:>16s}\n"
            else:
                dynamics_info += f"  Charge Embedding         = {'No':>16s}\n"
            # Print vdw interaction in MM program
            if (mm.vdw != None):
                dynamics_info += f"  VDW Interaction          = {mm.vdw:>16s}\n"
            else:
                dynamics_info += f"  VDW Interaction          = {'No':>16s}\n"
        
        dynamics_info += textwrap.indent(textwrap.dedent(f"""\

          MQC Method               = {self.md_type:>16s}
          Classical Path Approx.   = {'Yes':>16s}
          Time Interval (fs)       = {self.dt / fs_to_au:16.6f}
          Initial State (0:GS)     = {self.istate:>16d}
          Nuclear Step             = {self.nsteps:>16d}
        """), "  ")

        if (self.md_type != "BOMD"):
            dynamics_info += f"  Electronic Step          = {self.nesteps:>16d}\n"
            dynamics_info += f"  Electronic Propagator    = {self.propagator:>16s}\n"
            dynamics_info += f"  Propagation Scheme       = {self.elec_object:>16s}\n"

        # Print ad-hoc decoherence variables
        if (self.md_type == "SH"):
            if (self.dec_correction != None):
                dynamics_info += f"\n  Decoherence Scheme       = {self.dec_correction:>16s}\n"
                if (self.dec_correction == "edc"):
                    dynamics_info += f"  Energy Constant          = {self.edc_parameter:>16.6f}\n"
        
        # SHXF not implemented in this version!
        if (self.md_type == "SHXF"):
            error_message = "Invalid md type!"
            error_vars = f"md_type = {self.md_type}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Print system information
        dynamics_info += f"\n  Output Frequency         = {self.out_freq:>16d}\n"
        dynamics_info += f"  Verbosity Level          = {self.verbosity:>16d}\n"

        print (dynamics_info, flush=True)

    def touch_file(self, unixmd_dir):
        """ Routine to write PyUNIxMD output files

            :param string unixmd_dir: Directory where MD output files are written
        """
        # Energy information file header
        tmp = f'{"#":5s}{"Step":9s}{"Kinetic(H)":15s}{"Potential(H)":15s}{"Total(H)":15s}' + \
            "".join([f'E({ist})(H){"":8s}' for ist in range(self.mol.nst)])
        typewriter(tmp, unixmd_dir, "MDENERGY", "w")

        # BO coefficents, densities file header
        if (self.elec_object == "density"):
            tmp = f'{"#":5s} Density Matrix: population Re; see the manual for detail orders'
            typewriter(tmp, unixmd_dir, "BOPOP", "w")
            tmp = f'{"#":5s} Density Matrix: coherence Re-Im; see the manual for detail orders'
            typewriter(tmp, unixmd_dir, "BOCOH", "w")
        elif (self.elec_object == "coefficient"):
            tmp = f'{"#":5s} BO State Coefficients: state Re-Im; see the manual for detail orders'
            typewriter(tmp, unixmd_dir, "BOCOEF", "w")
            if (self.l_print_dm):
                tmp = f'{"#":5s} Density Matrix: population Re; see the manual for detail orders'
                typewriter(tmp, unixmd_dir, "BOPOP", "w")
                tmp = f'{"#":5s} Density Matrix: coherence Re-Im; see the manual for detail orders'
                typewriter(tmp, unixmd_dir, "BOCOH", "w")

        # NACME file header
        tmp = f'{"#":5s}Non-Adiabatic Coupling Matrix Elements: off-diagonal'
        typewriter(tmp, unixmd_dir, "NACME", "w")

        # DOTPOPNAC file header
        if (self.verbosity >= 1):
            tmp = f'{"#":5s} Time-derivative Density Matrix by NAC: population; see the manual for detail orders'
            typewriter(tmp, unixmd_dir, "DOTPOPNAC", "w")

        # file header for SH-based methods
        if (self.md_type in ["SH", "SHXF"]):
            tmp = f'{"#":5s}{"Step":8s}{"Running State":10s}'
            typewriter(tmp, unixmd_dir, "SHSTATE", "w")

            tmp = f'{"#":5s}{"Step":12s}' + "".join([f'Prob({ist}){"":8s}' for ist in range(self.mol.nst)])
            typewriter(tmp, unixmd_dir, "SHPROB", "w")

        # file header for XF-based methods
        if (self.md_type in ["SHXF"]):
            if (self.verbosity >= 1):
                tmp = f'{"#":5s} Time-derivative Density Matrix by decoherence: population; see the manual for detail orders'

    def write_md_output(self, unixmd_dir, istep):
        """ Write output files

            :param string unixmd_dir: Directory where MD output files are written
            :param integer istep: Current MD step
        """
        # Write MOVIE.xyz file including positions and velocities
        tmp = f'{self.mol.nat:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Position(A){"":34s}Velocity(au)' + \
            "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
            "".join([f'{self.mol.pos[iat, isp] * au_to_A:15.8f}' for isp in range(self.mol.ndim)]) + \
            "".join([f"{self.mol.vel[iat, isp]:15.8f}" for isp in range(self.mol.ndim)]) for iat in range(self.mol.nat)])
        typewriter(tmp, unixmd_dir, "MOVIE.xyz", "a")

        # Write MDENERGY file including several energy information
        tmp = f'{istep + 1:9d}{self.mol.ekin:15.8f}{self.mol.epot:15.8f}{self.mol.etot:15.8f}' \
            + "".join([f'{states.energy:15.8f}' for states in self.mol.states])
        typewriter(tmp, unixmd_dir, "MDENERGY", "a")

        # Write BOCOEF, BOPOP, BOCOH files
        if (self.elec_object == "density"):
            tmp = f'{istep + 1:9d}' + "".join([f'{self.mol.rho.real[ist, ist]:15.8f}' for ist in range(self.mol.nst)])
            typewriter(tmp, unixmd_dir, "BOPOP", "a")
            tmp = f'{istep + 1:9d}' + "".join([f"{self.mol.rho.real[ist, jst]:15.8f}{self.mol.rho.imag[ist, jst]:15.8f}" \
                for ist in range(self.mol.nst) for jst in range(ist + 1, self.mol.nst)])
            typewriter(tmp, unixmd_dir, "BOCOH", "a")
        elif (self.elec_object == "coefficient"):
            tmp = f'{istep + 1:9d}' + "".join([f'{states.coef.real:15.8f}{states.coef.imag:15.8f}' \
                for states in self.mol.states])
            typewriter(tmp, unixmd_dir, "BOCOEF", "a")
            if (self.l_print_dm):
                tmp = f'{istep + 1:9d}' + "".join([f'{self.mol.rho.real[ist, ist]:15.8f}' for ist in range(self.mol.nst)])
                typewriter(tmp, unixmd_dir, "BOPOP", "a")
                tmp = f'{istep + 1:9d}' + "".join([f"{self.mol.rho.real[ist, jst]:15.8f}{self.mol.rho.imag[ist, jst]:15.8f}" \
                    for ist in range(self.mol.nst) for jst in range(ist + 1, self.mol.nst)])

        # Write NACME file
        tmp = f'{istep + 1:10d}' + "".join([f'{self.mol.nacme[ist, jst]:15.8f}' \
            for ist in range(self.mol.nst) for jst in range(ist + 1, self.mol.nst)])
        typewriter(tmp, unixmd_dir, "NACME", "a")

        # Write NACV file
        if (not self.mol.l_nacme and self.verbosity >= 2):
            for ist in range(self.mol.nst):
                for jst in range(ist + 1, self.mol.nst):
                    tmp = f'{self.mol.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}NACV' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.mol.nac[ist, jst, iat, isp]:15.8f}' for isp in range(self.mol.ndim)]) for iat in range(self.mol.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"NACV_{ist}_{jst}", "a")

