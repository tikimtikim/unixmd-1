from __future__ import division
from cpa.cpa import CPA
from misc import eps, au_to_K, call_name, typewriter
import os, shutil, textwrap
import numpy as np
import pickle
import time

class BOMD(CPA):
    """ Class for born-oppenheimer molecular dynamics (BOMD) sampling
        to run dynamics with Classical Path Approximation (CPA)

        :param object molecule: Molecule object
        :param object thermostat: Thermostat object
        :param integer istate: Electronic state
        :param double dt: Time interval
        :param integer nsteps: Total step of nuclear propagation
        :param string unit_dt: Unit of time step
        :param integer out_freq: Frequency of printing output
        :param integer verbosity: Verbosity of output
        :param string samp_dir: Path of sampling data folder
    """
    def __init__(self, molecule, thermostat=None, istate=0, dt=0.5, nsteps=1000, \
        unit_dt="fs", out_freq=1, verbosity=0, samp_dir="./Data"):
        # Initialize input values
        super().__init__(molecule, istate, dt, nsteps, None, None, None, \
            False, None, None, unit_dt, out_freq, verbosity)

        self.thermo = thermostat
        self.samp_dir = samp_dir
        self.rforce = np.zeros((self.mol.nat, self.mol.ndim))

        if(not os.path.exists(self.samp_dir)):
            os.makedirs(self.samp_dir)
        else:
            error_message = "File already exists!"
            error_vars = f"samp_dir = {self.samp_dir}"
            raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

    def run(self, qm, mm=None, output_dir="./", l_save_qm_log=False, l_save_mm_log=False, l_save_scr=True, restart=None):
        """ Run BOMD to obtain binary for CPA dynamics

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
        bo_list = [self.istate]
        qm.calc_coupling = True
        self.print_init(qm, mm, restart)

        if (restart == None):
            # Calculate initial input geometry at t = 0.0 s
            self.istep = -1
            self.mol.reset_bo(qm.calc_coupling)
            qm.get_data(self.mol, base_dir, bo_list, self.dt, self.istep, calc_force_only=False)
            if (self.mol.l_qmmm and mm != None):
                mm.get_data(self.mol, base_dir, bo_list, self.istep, calc_force_only=False)
            self.update_energy()
            self.save_bin(self.istep)
            self.print_step(self.istep)

        elif (restart == "write"):
            # Reset initial time step to t = 0.0 s
            self.istep = -1
            self.write_md_output(unixmd_dir, self.istep)
            self.print_step(self.istep)

        elif (restart == "append"):
            # Set initial time step to last successful step of previous dynamics
            self.istep = self.fstep

        self.istep += 1

        # Main MD loop
        for istep in range(self.istep, self.nsteps):

            self.calculate_force()
            self.cl_update_position()
            
            self.mol.backup_bo()
            self.mol.reset_bo(qm.calc_coupling)
            qm.get_data(self.mol, base_dir, bo_list, self.dt, istep, calc_force_only=False)
            if (not self.mol.l_nacme):
                self.mol.get_nacme()
            if (self.mol.l_qmmm and mm != None):
                mm.get_data(self.mol, base_dir, bo_list, istep, calc_force_only=False)

            if (not self.mol.l_nacme):
                self.mol.adjust_nac()

            self.calculate_force()
            self.cl_update_velocity()

            if (self.thermo != None):
                self.thermo.run(self)
            
            self.update_energy()
            
            self.save_bin(istep)

            if ((istep + 1) % self.out_freq == 0):
                self.write_md_output(unixmd_dir, istep)
                self.print_step(istep)

            self.fstep = istep
            restart_file = os.path.join(base_dir, "RESTART.bin")
            with open(restart_file, 'wb') as f:
                pickle.dump({'qm':qm, 'md':self}, f)

        # Delete scratch directory
        if (not l_save_scr):
            tmp_dir = os.path.join(unixmd_dir, "scr_qm")
            if (os.path.exists(tmp_dir)):
                shutil.rmtree(tmp_dir)

            if (self.mol.l_qmmm and mm != None):
                tmp_dir = os.path.join(unixmd_dir, "scr_mm")
                if (os.path.exists(tmp_dir)):
                    shutil.rmtree(tmp_dir)

    def save_bin(self, istep):
        """ Routine to save MD info of each step using pickle
            
            :param integer istep: Current MD step
        """
        filename = os.path.join(self.samp_dir, "QM." + str(istep) + ".bin")
        with open(filename, "wb") as f:
            pickle.dump({"energy":np.array([x.energy for x in self.mol.states]), "force":self.rforce, "nacme":self.mol.nacme}, f)

        filename = os.path.join(self.samp_dir, "RP." + str(istep) + ".bin")
        with open(filename, "wb") as f:
            pickle.dump({"pos":self.mol.pos, "vel":self.mol.vel}, f)

    def calculate_force(self):
        """ Routine to calculate the forces
        """
        self.rforce = np.copy(self.mol.states[self.istate].force)

    def update_energy(self):
        """ Routine to update the energy of molecules in BOMD
        """
        # Update kinetic energy
        self.mol.update_kinetic()
        self.mol.epot = self.mol.states[self.istate].energy
        self.mol.etot = self.mol.epot + self.mol.ekin

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
        INIT = f" #INFO{'STEP':>8s}{'State':>7s}{'Kinetic(H)':>13s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}{'Norm.':>8s}"
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
        ctemp = self.mol.ekin * 2. / float(self.mol.ndof) * au_to_K
        norm = 0.
        for ist in range(self.mol.nst):
            norm += self.mol.rho.real[ist, ist]

        # Print INFO for each step
        INFO = f" INFO{istep + 1:>9d}{self.istate:>5d}"
        INFO += f"{self.mol.ekin:14.8f}{self.mol.epot:15.8f}{self.mol.etot:15.8f}"
        INFO += f"{ctemp:13.6f}"
        INFO += f"{norm:11.5f}"
        print (INFO, flush=True)

        # Print DEBUG1 for each step
        if (self.verbosity >= 1):
            cnt = 0
            if (self.l_mult_el_hop):
                DEBUG1 = ""
                occ_list = np.where(self.rocc_old == 1)[0]
                for ihop, iocc in enumerate(occ_list):
                    DEBUG1 += f" DEBUG1{istep + 1:>7d}"
                    DEBUG1 += f"{self.rand[ihop]:11.5f}"
                    for ist in range(self.mol.nst):
                        DEBUG1 += f"{self.acc_prob[ihop, ist]:12.5f} ({iocc + 1}->{ist + 1})"
                    DEBUG1 += "\n"
            else:
                DEBUG1 = f" DEBUG1{istep + 1:>7d}"
                DEBUG1 += f"{self.rand:11.5f}"
                for ist in range(self.mol.nst):
                    for jst in range(self.mol.nst):
                        cnt += 1
                        if (self.rocc_old[ist] == 0):
                            continue
                        DEBUG1 += f"{self.acc_prob[cnt]:12.5f} ({ist + 1}->{jst + 1})"
            print (DEBUG1, flush=True)
