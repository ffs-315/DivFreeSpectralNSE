import time as timer
from pathlib import Path

import numpy as np
from mpi4py import MPI
from shenfun import Array, Function, FunctionSpace, TensorProductSpace


program_start = timer.perf_counter()


# ============================================================
# Adaptive PC-CN simulation of the classical doubly periodic
# double-shear-layer problem on (0, 2*pi)^2.
#
# A scalar Fourier streamfunction is the only spatial unknown:
#     u = (partial_y psi, -partial_x psi).
# Hence the velocity is divergence-free by construction.  The
# original convective form (u dot grad)u is evaluated on a 3/2-padded
# physical grid, and every linear PC-CN solve is diagonal in Fourier
# space.  No pressure recovery is required.
# ============================================================
N = 256
rho = np.pi/15.0
perturbation_amplitude = 0.05
Reynolds_number = 1.0e4
nu = 1.0/Reynolds_number
final_time = 10.0

# Adaptive time-step parameters.
tau_min = 1.0e-4
tau_max = 1.0e-2
increment_tolerance = 1.0e-2
variation_limit = 5.0e-1
rho_safety = 0.9
r_max = 2.0
threshold = 20

# The state at the accepted time closest to each requested time is
# saved.  The accepted steps are not shortened merely to hit a plot
# time, so the adaptive dynamics are not disturbed.
vorticity_snapshot_times = np.array(
    [0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    dtype=float,
)

# Print progress every this many accepted steps.  Use None to disable.
progress_output_interval = 100
energy_residual_tolerance = 1.0e-11
maximum_accepted_steps = 2_000_000
maximum_rejected_trials = 2_000_000

data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/double_shear_flow"
)


if not isinstance(N, (int, np.integer)) or N < 8 or N % 2:
    raise ValueError("N must be an even integer no smaller than 8.")
if rho <= 0.0:
    raise ValueError("rho must be positive.")
if Reynolds_number <= 0.0:
    raise ValueError("Reynolds_number must be positive.")
if not (0.0 < tau_min <= tau_max):
    raise ValueError("Require 0 < tau_min <= tau_max.")
if increment_tolerance <= 0.0 or variation_limit <= 0.0:
    raise ValueError("The adaptive tolerances must be positive.")
if not (0.0 < rho_safety < 1.0):
    raise ValueError("rho_safety must lie in (0, 1).")
if r_max < 1.0:
    raise ValueError("r_max must satisfy r_max >= 1.")
if not isinstance(threshold, (int, np.integer)) or threshold <= 0:
    raise ValueError("threshold must be a positive integer.")
if progress_output_interval is not None and (
    not isinstance(progress_output_interval, (int, np.integer))
    or progress_output_interval <= 0
):
    raise ValueError(
        "progress_output_interval must be a positive integer or None."
    )
if energy_residual_tolerance <= 0.0:
    raise ValueError("energy_residual_tolerance must be positive.")
if vorticity_snapshot_times.ndim != 1:
    raise ValueError("vorticity_snapshot_times must be one-dimensional.")
if np.any(np.diff(vorticity_snapshot_times) <= 0.0):
    raise ValueError("vorticity_snapshot_times must be strictly increasing.")
if vorticity_snapshot_times.size and (
    vorticity_snapshot_times[0] < 0.0
    or vorticity_snapshot_times[-1] > final_time
):
    raise ValueError("Snapshot times must lie in [0, final_time].")


class EnergyCorrectionFailure(RuntimeError):
    """Raised when the supplementary-variable equation cannot be solved."""


class PeriodicDoubleShearDiscretization:
    def __init__(self, N, nu, rho, perturbation_amplitude, comm):
        self.N = int(N)
        self.nu = float(nu)
        self.rho = float(rho)
        self.perturbation_amplitude = float(perturbation_amplitude)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.volume = (2.0*np.pi)**2
        self.number_of_grid_points = self.N**2

        # Only the last transformed direction may use a real-to-complex
        # Fourier basis.  The resulting physical arrays remain real.
        Kx = FunctionSpace(
            self.N,
            "Fourier",
            dtype="D",
            domain=(0.0, 2.0*np.pi),
        )
        Ky = FunctionSpace(
            self.N,
            "Fourier",
            dtype="d",
            domain=(0.0, 2.0*np.pi),
        )
        self.T = TensorProductSpace(
            self.comm,
            (Kx, Ky),
            planner_effort="FFTW_MEASURE",
        )
        self.Tp = self.T.get_dealiased(padding_factor=1.5)

        wavenumbers = self.T.local_wavenumbers(
            broadcast=True,
            scaled=True,
            eliminate_highest_freq=False,
        )
        self.kx = np.asarray(wavenumbers[0])
        self.ky = np.asarray(wavenumbers[1])
        self.k_squared = self.kx**2 + self.ky**2
        self.nonzero_mode = self.k_squared > 0.0
        self.nyquist_mask = self.T.get_mask_nyquist()

        # Work arrays on the base grid.
        self.base_1 = Array(self.T)
        self.base_2 = Array(self.T)
        self.base_3 = Array(self.T)
        self.base_4 = Array(self.T)

        # Work arrays on the 3/2-padded grid.  All six fields must coexist
        # while the two components of (u dot grad)u are formed.
        self.pad_u1 = Array(self.Tp)
        self.pad_u2 = Array(self.Tp)
        self.pad_u1_x = Array(self.Tp)
        self.pad_u1_y = Array(self.Tp)
        self.pad_u2_x = Array(self.Tp)
        self.pad_u2_y = Array(self.Tp)
        self.nonlinear_1_hat = Function(self.T)
        self.nonlinear_2_hat = Function(self.T)
        self.streamfunction_rhs_hat = Function(self.T)

    def new_function(self, values=None):
        result = Function(self.T)
        if values is not None:
            result[:] = values
        return result

    def apply_spectral_mask(self, values):
        if self.nyquist_mask is not None:
            values *= self.nyquist_mask
        values[~self.nonzero_mode] = 0.0
        return values

    def initial_streamfunction(self):
        x, y = self.T.local_mesh(True)
        velocity_1 = Array(self.T)
        velocity_2 = Array(self.T)
        velocity_1[:] = np.where(
            y <= np.pi,
            np.tanh((y - 0.5*np.pi)/self.rho),
            np.tanh((1.5*np.pi - y)/self.rho),
        )
        velocity_2[:] = self.perturbation_amplitude*np.sin(x)

        velocity_1_hat = self.T.forward(
            velocity_1,
            Function(self.T),
        )
        velocity_2_hat = self.T.forward(
            velocity_2,
            Function(self.T),
        )

        # u_1 = i*k_y*psi and u_2 = -i*k_x*psi.  This orthogonal
        # reconstruction removes roundoff-level non-solenoidal content
        # and the (analytically zero) mean velocity.
        denominator = np.where(
            self.nonzero_mode,
            self.k_squared,
            1.0,
        )
        streamfunction = self.new_function()
        streamfunction[:] = (
            -1j*self.ky*velocity_1_hat
            + 1j*self.kx*velocity_2_hat
        )/denominator
        self.apply_spectral_mask(streamfunction)
        return streamfunction

    def nonlinear_streamfunction_rhs(self, streamfunction):
        # Original convective form N=(u dot grad)u evaluated on the
        # 3/2-padded physical mesh.
        self.Tp.backward(
            1j*self.ky*streamfunction,
            self.pad_u1,
        )
        self.Tp.backward(
            -1j*self.kx*streamfunction,
            self.pad_u2,
        )
        self.Tp.backward(
            -self.kx*self.ky*streamfunction,
            self.pad_u1_x,
        )
        self.Tp.backward(
            -self.ky**2*streamfunction,
            self.pad_u1_y,
        )
        self.Tp.backward(
            self.kx**2*streamfunction,
            self.pad_u2_x,
        )
        self.Tp.backward(
            self.kx*self.ky*streamfunction,
            self.pad_u2_y,
        )

        nonlinear_1 = (
            self.pad_u1*self.pad_u1_x
            + self.pad_u2*self.pad_u1_y
        )
        nonlinear_2 = (
            self.pad_u1*self.pad_u2_x
            + self.pad_u2*self.pad_u2_y
        )
        self.Tp.forward(nonlinear_1, self.nonlinear_1_hat)
        self.Tp.forward(nonlinear_2, self.nonlinear_2_hat)

        # curl N = partial_x N_2 - partial_y N_1, whereas the right-hand
        # side of the streamfunction equation is -curl N.
        self.streamfunction_rhs_hat[:] = (
            -1j*self.kx*self.nonlinear_2_hat
            + 1j*self.ky*self.nonlinear_1_hat
        )
        self.apply_spectral_mask(self.streamfunction_rhs_hat)
        return self.streamfunction_rhs_hat

    def solve_pc_cn_linear_system(
        self,
        previous_streamfunction,
        nonlinear_rhs,
        time_step,
    ):
        gamma = 0.5*self.nu*float(time_step)
        denominator = (
            self.k_squared*(1.0 + gamma*self.k_squared)
        )
        safe_denominator = np.where(
            self.nonzero_mode,
            denominator,
            1.0,
        )
        result = self.new_function()
        result[:] = (
            self.k_squared*(1.0 - gamma*self.k_squared)
            * previous_streamfunction
            + float(time_step)*nonlinear_rhs
        )/safe_denominator
        self.apply_spectral_mask(result)
        return result

    def physical_integral(self, values):
        local_sum = float(np.sum(np.asarray(values, dtype=float)))
        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        return self.volume*global_sum/self.number_of_grid_points

    def velocity_values(self, streamfunction):
        self.T.backward(
            1j*self.ky*streamfunction,
            self.base_1,
        )
        self.T.backward(
            -1j*self.kx*streamfunction,
            self.base_2,
        )
        return self.base_1, self.base_2

    def energy(self, streamfunction):
        velocity_1, velocity_2 = self.velocity_values(streamfunction)
        return 0.5*self.physical_integral(
            velocity_1**2 + velocity_2**2
        )

    def l2_norm(self, streamfunction):
        return np.sqrt(max(2.0*self.energy(streamfunction), 0.0))

    def gradient_norm_squared(self, streamfunction):
        self.T.backward(
            -self.kx*self.ky*streamfunction,
            self.base_1,
        )
        self.T.backward(
            -self.ky**2*streamfunction,
            self.base_2,
        )
        self.T.backward(
            self.kx**2*streamfunction,
            self.base_3,
        )
        self.T.backward(
            self.kx*self.ky*streamfunction,
            self.base_4,
        )
        return self.physical_integral(
            self.base_1**2
            + self.base_2**2
            + self.base_3**2
            + self.base_4**2
        )

    def divergence_norm(self, streamfunction):
        # partial_x u_1 and partial_y u_2 are transformed separately so
        # this diagnostic also detects implementation and transform errors.
        self.T.backward(
            -self.kx*self.ky*streamfunction,
            self.base_1,
        )
        self.T.backward(
            self.kx*self.ky*streamfunction,
            self.base_2,
        )
        return np.sqrt(
            max(
                self.physical_integral((self.base_1 + self.base_2)**2),
                0.0,
            )
        )

    def vorticity_values(self, streamfunction):
        # omega = partial_x u_2 - partial_y u_1 = -Delta psi.
        self.T.backward(
            self.k_squared*streamfunction,
            self.base_1,
        )
        return self.base_1

    def vorticity_diagnostics(self, streamfunction):
        vorticity = self.vorticity_values(streamfunction)
        enstrophy = 0.5*self.physical_integral(vorticity**2)
        local_maximum = float(np.max(np.abs(vorticity)))
        maximum = self.comm.allreduce(local_maximum, op=MPI.MAX)
        return enstrophy, maximum

    def gather_distributed_array(self, values, forward_output):
        local_slice = self.T.local_slice(forward_output)
        local_values = np.asarray(values).copy()
        pieces = self.comm.gather(
            (local_slice, local_values),
            root=0,
        )
        if self.rank != 0:
            return None

        global_values = np.empty(
            self.T.global_shape(forward_output),
            dtype=local_values.dtype,
        )
        for array_slice, array_values in pieces:
            global_values[array_slice] = array_values
        return global_values

    def gather_snapshot(self, streamfunction):
        vorticity = self.vorticity_values(streamfunction)
        global_vorticity = self.gather_distributed_array(
            vorticity,
            forward_output=False,
        )
        global_streamfunction = self.gather_distributed_array(
            streamfunction,
            forward_output=True,
        )
        if self.rank == 0:
            # Shenfun stores physical arrays as [x,y].  The transpose is
            # saved so plotting routines receive the conventional [y,x].
            global_vorticity = global_vorticity.T
        return global_vorticity, global_streamfunction

    def supplementary_update(
        self,
        provisional_streamfunction,
        previous_streamfunction,
        energy_streamfunction,
        time_step,
    ):
        energy_previous = self.energy(previous_streamfunction)
        energy_before = self.energy(provisional_streamfunction)
        dissipation = (
            self.nu*self.gradient_norm_squared(energy_streamfunction)
        )

        # For a periodic problem there is no lifting and no wall work.
        # E((1+beta)u_1)=(1+beta)^2 E(u_1).
        coefficient_a = energy_before
        coefficient_b = 2.0*energy_before
        coefficient_c = (
            energy_before
            - energy_previous
            + float(time_step)*dissipation
        )

        discriminant_scale = max(
            coefficient_b**2,
            abs(4.0*coefficient_a*coefficient_c),
            np.finfo(float).tiny,
        )
        discriminant = (
            coefficient_b**2
            - 4.0*coefficient_a*coefficient_c
        )
        discriminant_tolerance = (
            100.0*np.finfo(float).eps*discriminant_scale
        )
        if discriminant < -discriminant_tolerance:
            raise EnergyCorrectionFailure(
                "The supplementary-variable quadratic has no real root: "
                f"discriminant={discriminant:.6e}, relative discriminant="
                f"{discriminant/discriminant_scale:.6e}."
            )
        discriminant = max(discriminant, 0.0)

        tiny = 1.0e-30
        if abs(coefficient_a) <= tiny:
            if abs(coefficient_b) <= tiny:
                if abs(coefficient_c) > 1.0e-12:
                    raise EnergyCorrectionFailure(
                        "The degenerate supplementary-variable equation "
                        "cannot satisfy the energy balance."
                    )
                beta = 0.0
            else:
                beta = -coefficient_c/coefficient_b
        else:
            square_root = np.sqrt(discriminant)
            q = -0.5*(
                coefficient_b
                + np.copysign(square_root, coefficient_b)
            )
            if abs(q) <= tiny:
                beta = -coefficient_b/(2.0*coefficient_a)
            else:
                roots = np.array(
                    [
                        q/coefficient_a,
                        coefficient_c/q,
                    ]
                )
                beta = float(roots[np.argmin(np.abs(roots))])

        corrected_streamfunction = self.new_function(
            (1.0 + beta)*provisional_streamfunction
        )
        self.apply_spectral_mask(corrected_streamfunction)
        alpha = beta/float(time_step)
        energy_after = self.energy(corrected_streamfunction)
        residual_pre = coefficient_c/float(time_step)
        residual_post = (
            (energy_after - energy_previous)/float(time_step)
            + dissipation
        )
        residual_scale = max(
            abs((energy_after - energy_previous)/float(time_step)),
            abs(dissipation),
            1.0,
        )
        # The residual contains a difference of two nearly equal kinetic
        # energies divided by tau_n.  Consequently, roundoff in the energy
        # evaluation is amplified by 1/tau_n when the time step is small.
        # Do not reject an otherwise valid supplementary-variable update
        # merely because this unavoidable roundoff floor is larger than the
        # requested relative residual tolerance.  The unmodified residual is
        # still returned and written to the history file.
        roundoff_floor = (
            200.0*np.finfo(float).eps
            * max(
                abs(energy_previous),
                abs(energy_before),
                abs(energy_after),
                1.0,
            )
            / float(time_step)
        )
        allowed_residual = max(
            energy_residual_tolerance*residual_scale,
            roundoff_floor,
        )
        if (
            not np.isfinite(residual_post)
            or abs(residual_post) > allowed_residual
        ):
            raise EnergyCorrectionFailure(
                "The supplementary-variable correction did not satisfy "
                "the periodic discrete energy equation: "
                f"R_post={residual_post:.6e}, scaled residual="
                f"{abs(residual_post)/residual_scale:.6e}, "
                f"allowed residual={allowed_residual:.6e}."
            )

        return {
            "corrected_streamfunction": corrected_streamfunction,
            "alpha": alpha,
            "energy_previous": energy_previous,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "dissipation": dissipation,
            "residual_pre": residual_pre,
            "residual_post": residual_post,
        }


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
discretization = PeriodicDoubleShearDiscretization(
    N=N,
    nu=nu,
    rho=rho,
    perturbation_amplitude=perturbation_amplitude,
    comm=comm,
)

if rank == 0:
    data_dir.mkdir(parents=True, exist_ok=True)
comm.Barrier()


history_columns = (
    "step",
    "time",
    "tau",
    "energy_before",
    "energy_after",
    "divergence_before",
    "divergence_after",
    "alpha",
    "residual_pre",
    "residual_post",
    "dissipation",
    "enstrophy",
    "maximum_absolute_vorticity",
    "delta",
    "variation_indicator",
    "rejected_trials_before_acceptance",
)


def compute_pc_cn_trial(
    streamfunction_previous,
    streamfunction_previous_previous,
    accepted_step_number,
    previous_time_step,
    trial_time_step,
):
    if accepted_step_number == 1:
        extrapolated_streamfunction = discretization.new_function(
            streamfunction_previous
        )
    else:
        time_step_ratio = trial_time_step/previous_time_step
        extrapolated_streamfunction = discretization.new_function(
            (1.0 + 0.5*time_step_ratio)*streamfunction_previous
            - 0.5*time_step_ratio*streamfunction_previous_previous
        )

    predicted_rhs = discretization.nonlinear_streamfunction_rhs(
        extrapolated_streamfunction
    )
    predicted_streamfunction = (
        discretization.solve_pc_cn_linear_system(
            streamfunction_previous,
            predicted_rhs,
            trial_time_step,
        )
    )

    predicted_midpoint_streamfunction = discretization.new_function(
        0.5*(predicted_streamfunction + streamfunction_previous)
    )
    corrected_rhs = discretization.nonlinear_streamfunction_rhs(
        predicted_midpoint_streamfunction
    )
    provisional_streamfunction = (
        discretization.solve_pc_cn_linear_system(
            streamfunction_previous,
            corrected_rhs,
            trial_time_step,
        )
    )

    update = discretization.supplementary_update(
        provisional_streamfunction,
        streamfunction_previous,
        predicted_midpoint_streamfunction,
        trial_time_step,
    )
    update["provisional_streamfunction"] = provisional_streamfunction
    return update


streamfunction_previous = discretization.initial_streamfunction()
streamfunction_previous_previous = discretization.new_function(
    streamfunction_previous
)
time_previous = 0.0
accepted_step_number = 1
previous_time_step = tau_min
proposed_time_step = tau_min
counter = 0
total_rejected_trials = 0
records = {name: [] for name in history_columns}
time_tolerance = 10.0*np.finfo(float).eps*max(1.0, final_time)

snapshot_target_times = []
snapshot_actual_times = []
snapshot_vorticities = []
snapshot_streamfunctions = []
next_snapshot_index = 0


def save_snapshot(target_time, actual_time, streamfunction):
    global_vorticity, global_streamfunction = (
        discretization.gather_snapshot(streamfunction)
    )
    if rank == 0:
        snapshot_target_times.append(float(target_time))
        snapshot_actual_times.append(float(actual_time))
        snapshot_vorticities.append(global_vorticity)
        snapshot_streamfunctions.append(global_streamfunction)


while (
    next_snapshot_index < vorticity_snapshot_times.size
    and vorticity_snapshot_times[next_snapshot_index] <= time_tolerance
):
    save_snapshot(
        vorticity_snapshot_times[next_snapshot_index],
        0.0,
        streamfunction_previous,
    )
    next_snapshot_index += 1


while time_previous < final_time - time_tolerance:
    if accepted_step_number > maximum_accepted_steps:
        raise RuntimeError("The maximum accepted-step count was exceeded.")

    trial_accepted = False
    rejected_before_acceptance = 0

    while not trial_accepted:
        # As in the cavity program, the final accepted step may pass T.
        # Snapshot states are selected afterwards from the nearest accepted
        # solution, so output requests do not create artificial tiny steps.
        trial_time_step = proposed_time_step

        try:
            trial = compute_pc_cn_trial(
                streamfunction_previous,
                streamfunction_previous_previous,
                accepted_step_number,
                previous_time_step,
                trial_time_step,
            )
        except EnergyCorrectionFailure:
            if trial_time_step <= tau_min*(1.0 + 1.0e-14):
                raise
            proposed_time_step = max(
                tau_min,
                rho_safety*0.5*trial_time_step,
            )
            counter = 0
            rejected_before_acceptance += 1
            total_rejected_trials += 1
            if total_rejected_trials > maximum_rejected_trials:
                raise RuntimeError(
                    "The maximum rejected-trial count was exceeded."
                )
            continue

        trial_streamfunction = trial["corrected_streamfunction"]
        difference = discretization.new_function(
            trial_streamfunction - streamfunction_previous
        )
        delta = discretization.l2_norm(difference)
        if not np.isfinite(delta):
            raise RuntimeError("The adaptive velocity increment is invalid.")

        if delta > increment_tolerance and trial_time_step > tau_min:
            proposed_time_step = max(
                tau_min,
                rho_safety
                * (increment_tolerance/delta)
                * trial_time_step,
            )
            counter = 0
            rejected_before_acceptance += 1
            total_rejected_trials += 1
            if total_rejected_trials > maximum_rejected_trials:
                raise RuntimeError(
                    "The maximum rejected-trial count was exceeded."
                )
            continue

        trial_accepted = True
        time_current = time_previous + trial_time_step
        variation_indicator = delta/trial_time_step
        provisional_streamfunction = trial["provisional_streamfunction"]
        enstrophy, maximum_absolute_vorticity = (
            discretization.vorticity_diagnostics(trial_streamfunction)
        )

        values = (
            accepted_step_number,
            time_current,
            trial_time_step,
            trial["energy_before"],
            trial["energy_after"],
            discretization.divergence_norm(provisional_streamfunction),
            discretization.divergence_norm(trial_streamfunction),
            trial["alpha"],
            trial["residual_pre"],
            trial["residual_post"],
            trial["dissipation"],
            enstrophy,
            maximum_absolute_vorticity,
            delta,
            variation_indicator,
            rejected_before_acceptance,
        )
        for name, value in zip(history_columns, values):
            records[name].append(value)

        if (
            progress_output_interval is not None
            and accepted_step_number % progress_output_interval == 0
            and rank == 0
        ):
            print(
                f"step={accepted_step_number:7d}, "
                f"t_n={time_current:.4f}, "
                f"tau_n={trial_time_step:.2e}, "
                f"e_n={variation_indicator:.2e}, "
                f"E(u^n)={trial['energy_after']:.6e}"
            )

        # Save the accepted state nearest to each requested snapshot time
        # crossed by the current step.
        while (
            next_snapshot_index < vorticity_snapshot_times.size
            and vorticity_snapshot_times[next_snapshot_index]
            <= time_current + time_tolerance
        ):
            target_time = vorticity_snapshot_times[next_snapshot_index]
            if (
                abs(target_time - time_previous)
                <= abs(time_current - target_time)
            ):
                snapshot_state = streamfunction_previous
                snapshot_time = time_previous
            else:
                snapshot_state = trial_streamfunction
                snapshot_time = time_current
            save_snapshot(target_time, snapshot_time, snapshot_state)
            next_snapshot_index += 1

        if variation_indicator < variation_limit:
            counter += 1
            if counter >= threshold:
                previous_norm = discretization.l2_norm(
                    streamfunction_previous
                )
                trial_norm = discretization.l2_norm(
                    trial_streamfunction
                )
                delta_floor = np.finfo(float).eps*max(
                    1.0,
                    previous_norm,
                    trial_norm,
                )
                if delta <= delta_floor:
                    rho_growth = r_max
                else:
                    rho_growth = max(
                        1.0,
                        min(
                            rho_safety*increment_tolerance/delta,
                            r_max,
                        ),
                    )
                next_time_step = min(
                    rho_growth*trial_time_step,
                    tau_max,
                )
                counter = 0
            else:
                next_time_step = trial_time_step
        else:
            next_time_step = trial_time_step
            counter = 0

        streamfunction_previous_previous = discretization.new_function(
            streamfunction_previous
        )
        streamfunction_previous = discretization.new_function(
            trial_streamfunction
        )
        previous_time_step = trial_time_step
        proposed_time_step = next_time_step
        time_previous = time_current
        accepted_step_number += 1


integer_fields = {
    "step",
    "rejected_trials_before_acceptance",
}
history = {
    name: np.asarray(
        records[name],
        dtype=int if name in integer_fields else float,
    )
    for name in history_columns
}
accepted_steps = int(history["step"].size)
run_elapsed = timer.perf_counter() - program_start


if rank == 0:
    history_npz_file = data_dir / "double_shear_pc_cn_history.npz"
    history_csv_file = data_dir / "double_shear_pc_cn_history.csv"
    snapshot_file = (
        data_dir / "double_shear_pc_cn_vorticity_snapshots.npz"
    )

    np.savez(
        history_npz_file,
        **history,
        N=N,
        rho=rho,
        perturbation_amplitude=perturbation_amplitude,
        Reynolds_number=Reynolds_number,
        nu=nu,
        requested_final_time=final_time,
        attained_final_time=history["time"][-1],
        tau_min=tau_min,
        tau_max=tau_max,
        increment_tolerance=increment_tolerance,
        variation_limit=variation_limit,
        rho_safety=rho_safety,
        r_max=r_max,
        threshold=threshold,
        dealiasing_rule="3/2 padding",
        convective_form="original (u dot grad)u",
        spatial_representation="Fourier streamfunction",
        accepted_steps=accepted_steps,
        rejected_trials=total_rejected_trials,
        elapsed_seconds=run_elapsed,
        energy_residual_tolerance=energy_residual_tolerance,
    )

    history_matrix = np.column_stack(
        [history[name] for name in history_columns]
    )
    history_formats = (
        ["%d"]
        + ["%.16e"]*14
        + ["%d"]
    )
    np.savetxt(
        history_csv_file,
        history_matrix,
        delimiter=",",
        header=",".join(history_columns),
        comments="",
        fmt=history_formats,
    )

    coordinates = np.linspace(
        0.0,
        2.0*np.pi,
        N,
        endpoint=False,
    )
    np.savez_compressed(
        snapshot_file,
        x=coordinates,
        y=coordinates.copy(),
        target_times=np.asarray(snapshot_target_times),
        actual_times=np.asarray(snapshot_actual_times),
        vorticity=np.stack(snapshot_vorticities, axis=0),
        streamfunction_hat=np.stack(
            snapshot_streamfunctions,
            axis=0,
        ),
        N=N,
        rho=rho,
        perturbation_amplitude=perturbation_amplitude,
        Reynolds_number=Reynolds_number,
        nu=nu,
    )

    print("\nAdaptive PC-CN doubly periodic double-shear-layer simulation")
    print(f"  N = {N}")
    print(f"  rho = {rho:.8e}")
    print(f"  perturbation amplitude = {perturbation_amplitude:.6g}")
    print(f"  Reynolds number = {Reynolds_number:.6g}")
    print(f"  attained final time = {history['time'][-1]:.8e}")
    print(f"  accepted steps = {accepted_steps}")
    print(f"  rejected trials = {total_rejected_trials}")
    print(f"  elapsed time = {run_elapsed:.2f} s")
    print(f"  maximum |alpha| = {np.max(np.abs(history['alpha'])):.6e}")
    print(
        "  maximum post-correction energy residual = "
        f"{np.max(np.abs(history['residual_post'])):.6e}"
    )
    print(
        "  maximum post-correction divergence = "
        f"{np.max(history['divergence_after']):.6e}"
    )
    print(f"  history NPZ: {history_npz_file}")
    print(f"  history CSV: {history_csv_file}")
    print(f"  vorticity snapshots: {snapshot_file}")
