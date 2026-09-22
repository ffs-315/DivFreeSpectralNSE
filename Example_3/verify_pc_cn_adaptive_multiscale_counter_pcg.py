import time as timer
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from example2_shenfun_common_pcg import Example2Discretization


program_start = timer.perf_counter()


# ============================================================
# Multiscale-in-time PC-CN adaptive-step experiment
# ============================================================
N = 24
nu = 1.0e-8
final_time = 50.0

number_of_windows = 3
omega = 2*np.pi
amplitude_scale = 0.20
window_left = np.array([15], dtype=float)
window_right = np.array([17], dtype=float)
transition_width = np.array([5], dtype=float)

tau_min = 1.0e-3
tau_max = 1.0e-1
increment_tolerance_values = (1.0e-2, 1.0e-2, 1.0e-2)
variation_limit_values = (1.0e-1, 5.0e-2, 3e-2)
rho_safety = 0.9
r_max = 2.0
threshold = 10

for parameter_name, parameter_values in (
    ("increment_tolerance_values", increment_tolerance_values),
    ("variation_limit_values", variation_limit_values),
):
    if len(parameter_values) == 0:
        raise ValueError(f"{parameter_name} must not be empty.")
    if any(value <= 0.0 for value in parameter_values):
        raise ValueError(f"Every entry of {parameter_name} must be positive.")

if len(increment_tolerance_values) != len(variation_limit_values):
    raise ValueError(
        "increment_tolerance_values and variation_limit_values must have "
        "the same length because their entries are paired by position."
    )

parameter_pairs = tuple(
    zip(increment_tolerance_values, variation_limit_values)
)

if not isinstance(threshold, (int, np.integer)) or threshold <= 0:
    raise ValueError("threshold must be a positive integer.")

pcg_tolerance = 1.0e-12
pcg_max_iterations = None
recover_pressure_after_acceptance = True
maximum_accepted_steps = 1_000_000
maximum_rejected_trials = 1_000_000

data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/test_counter_time_algorithm"
)
data_dir.mkdir(parents=True, exist_ok=True)


class MultiscaleTimeDiscretization(Example2Discretization):
    def __init__(
        self,
        N,
        nu,
        final_time,
        omega,
        amplitude_scale,
        window_left,
        window_right,
        transition_width,
        pcg_tolerance,
        pcg_max_iterations,
    ):
        self.final_time = float(final_time)
        self.omega = float(omega)
        self.amplitude_scale = float(amplitude_scale)
        self.window_left = np.asarray(window_left, dtype=float)
        self.window_right = np.asarray(window_right, dtype=float)
        self.transition_width = np.asarray(
            transition_width,
            dtype=float,
        )

        if not (
            self.window_left.size
            == self.window_right.size
            == self.transition_width.size
        ):
            raise ValueError("The temporal-window arrays must have equal size.")
        if np.any(self.window_left >= self.window_right):
            raise ValueError("Each temporal window must satisfy a_i < b_i.")
        if np.any(self.transition_width <= 0.0):
            raise ValueError("Every transition width must be positive.")
        if self.amplitude_scale <= 0.0:
            raise ValueError("amplitude_scale must be positive.")

        self.rho_shift = self._compute_nonnegative_shift()

        super().__init__(
            N=N,
            nu=nu,
            pcg_tolerance=pcg_tolerance,
            pcg_max_iterations=pcg_max_iterations,
        )

    def rho_unshifted(self, time):
        time_array = np.asarray(time, dtype=float)
        window_sum = np.zeros_like(time_array, dtype=float)

        for left, right, width in zip(
            self.window_left,
            self.window_right,
            self.transition_width,
        ):
            window_sum += (
                np.tanh((time_array - left)/width)
                - np.tanh((time_array - right)/width)
            )

        values = (
            self.amplitude_scale
            * np.sin(self.omega*time_array)
            * window_sum
        )
        return float(values) if values.ndim == 0 else values

    def rho_derivative(self, time):
        time_array = np.asarray(time, dtype=float)
        window_sum = np.zeros_like(time_array, dtype=float)
        window_derivative_sum = np.zeros_like(time_array, dtype=float)

        for left, right, width in zip(
            self.window_left,
            self.window_right,
            self.transition_width,
        ):
            left_argument = (time_array - left)/width
            right_argument = (time_array - right)/width
            left_tanh = np.tanh(left_argument)
            right_tanh = np.tanh(right_argument)

            window_sum += left_tanh - right_tanh
            window_derivative_sum += (
                (1.0 - left_tanh**2)/width
                - (1.0 - right_tanh**2)/width
            )

        values = self.amplitude_scale*(
            self.omega*np.cos(self.omega*time_array)*window_sum
            + np.sin(self.omega*time_array)*window_derivative_sum
        )
        return float(values) if values.ndim == 0 else values

    def _compute_nonnegative_shift(self):
        sample_times = np.linspace(0.0, self.final_time, 200_001)
        sample_values = self.rho_unshifted(sample_times)
        minimum_index = int(np.argmin(sample_values))

        left_index = max(minimum_index - 2, 0)
        right_index = min(minimum_index + 2, sample_times.size - 1)
        local_left = float(sample_times[left_index])
        local_right = float(sample_times[right_index])

        minimum_value = float(sample_values[minimum_index])
        if local_right > local_left:
            result = minimize_scalar(
                self.rho_unshifted,
                bounds=(local_left, local_right),
                method="bounded",
                options={"xatol": 1.0e-14},
            )
            if result.success:
                minimum_value = min(minimum_value, float(result.fun))

        minimum_value = min(
            minimum_value,
            float(self.rho_unshifted(0.0)),
            float(self.rho_unshifted(self.final_time)),
        )
        return -minimum_value

    def rho(self, time):
        return self.rho_unshifted(time) + self.rho_shift

    def exact_fields(self, time):
        x = self.X
        y = self.Y
        temporal_factor = self.rho(time)

        u1 = (
            temporal_factor
            * np.sin(np.pi*x)**2
            * np.sin(2.0*np.pi*y)
        )
        u2 = (
            -temporal_factor
            * np.sin(2.0*np.pi*x)
            * np.sin(np.pi*y)**2
        )
        pressure = (
            temporal_factor
            * np.cos(np.pi*x)
            * np.sin(np.pi*y)
        )
        return u1, u2, pressure

    def exact_velocity_derivatives(self, time):
        x = self.X
        y = self.Y
        temporal_factor = self.rho(time)

        u1_x = (
            temporal_factor*np.pi
            * np.sin(2.0*np.pi*x)
            * np.sin(2.0*np.pi*y)
        )
        u1_y = (
            2.0*temporal_factor*np.pi
            * np.sin(np.pi*x)**2
            * np.cos(2.0*np.pi*y)
        )
        u2_x = (
            -2.0*temporal_factor*np.pi
            * np.cos(2.0*np.pi*x)
            * np.sin(np.pi*y)**2
        )
        u2_y = (
            -temporal_factor*np.pi
            * np.sin(2.0*np.pi*x)
            * np.sin(2.0*np.pi*y)
        )
        return u1_x, u1_y, u2_x, u2_y

    def forcing_values(self, time):
        x = self.X
        y = self.Y
        temporal_factor = self.rho(time)
        temporal_derivative = self.rho_derivative(time)

        spatial_u1 = np.sin(np.pi*x)**2*np.sin(2.0*np.pi*y)
        spatial_u2 = -np.sin(2.0*np.pi*x)*np.sin(np.pi*y)**2

        u1 = temporal_factor*spatial_u1
        u2 = temporal_factor*spatial_u2
        u1_x, u1_y, u2_x, u2_y = (
            self.exact_velocity_derivatives(time)
        )

        convection_1 = u1*u1_x + u2*u1_y
        convection_2 = u1*u2_x + u2*u2_y

        laplace_u1 = temporal_factor*(
            2.0*np.pi**2*np.cos(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
            - 4.0*np.pi**2*np.sin(np.pi*x)**2*np.sin(2.0*np.pi*y)
        )
        laplace_u2 = temporal_factor*(
            4.0*np.pi**2*np.sin(2.0*np.pi*x)*np.sin(np.pi*y)**2
            - 2.0*np.pi**2*np.sin(2.0*np.pi*x)*np.cos(2.0*np.pi*y)
        )

        pressure_x = (
            -temporal_factor*np.pi
            * np.sin(np.pi*x)*np.sin(np.pi*y)
        )
        pressure_y = (
            temporal_factor*np.pi
            * np.cos(np.pi*x)*np.cos(np.pi*y)
        )

        forcing_1 = (
            temporal_derivative*spatial_u1
            - self.nu*laplace_u1
            + convection_1
            + pressure_x
        )
        forcing_2 = (
            temporal_derivative*spatial_u2
            - self.nu*laplace_u2
            + convection_2
            + pressure_y
        )
        return forcing_1, forcing_2

    def initial_velocity_coefficients(self):
        return self.rho(0.0)*super().initial_velocity_coefficients()

    def exact_velocity_l2_norm(self, time):
        exact_1, exact_2, _ = self.exact_fields(time)
        return np.sqrt(
            np.sum(
                self.w[:, None]
                * (exact_1**2 + exact_2**2)
                * self.w[None, :]
            )
        )

    def numerical_velocity_l2_norm(self, coefficients):
        return np.sqrt(max(2.0*self.energy(coefficients), 0.0))

    def coefficient_l2_norm(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        squared_norm = float(
            coefficients @ self.apply_velocity_mass(coefficients)
        )
        return np.sqrt(max(squared_norm, 0.0))


discretization = MultiscaleTimeDiscretization(
    N=N,
    nu=nu,
    final_time=final_time,
    omega=omega,
    amplitude_scale=amplitude_scale,
    window_left=window_left,
    window_right=window_right,
    transition_width=transition_width,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=pcg_max_iterations,
)


column_names = (
    "step",
    "time",
    "tau",
    "exact_velocity_l2",
    "numerical_velocity_l2",
    "velocity_l2_error",
    "alpha",
    "delta",
    "variation_indicator",
    "rejected_trials_before_acceptance",
    "pcg_iterations_prediction",
    "pcg_iterations_correction",
)


def compute_pc_cn_trial(
    velocity_previous,
    velocity_previous_previous,
    accepted_step_number,
    previous_time_step,
    trial_time_step,
    midpoint_time,
):
    if accepted_step_number == 1:
        extrapolated_velocity = velocity_previous.copy()
    else:
        time_step_ratio = trial_time_step/previous_time_step
        extrapolated_velocity = (
            (1.0 + 0.5*time_step_ratio)*velocity_previous
            - 0.5*time_step_ratio*velocity_previous_previous
        )

    forcing_load = discretization.forcing_load(midpoint_time)
    common_right_hand_side = (
        discretization.apply_velocity_mass(velocity_previous)
        - 0.5*nu*trial_time_step
        * discretization.apply_velocity_stiffness(velocity_previous)
    )

    predicted_right_hand_side = (
        common_right_hand_side
        + trial_time_step*(
            forcing_load
            - discretization.convection_load(extrapolated_velocity)
        )
    )
    predicted_velocity = discretization.solve_velocity(
        trial_time_step,
        predicted_right_hand_side,
        initial_guess=velocity_previous,
    )
    prediction_iterations = discretization.last_pcg_iterations

    predicted_midpoint_velocity = 0.5*(
        predicted_velocity + velocity_previous
    )

    corrected_right_hand_side = (
        common_right_hand_side
        + trial_time_step*(
            forcing_load
            - discretization.convection_load(predicted_midpoint_velocity)
        )
    )
    provisional_velocity = discretization.solve_velocity(
        trial_time_step,
        corrected_right_hand_side,
        initial_guess=predicted_velocity,
    )
    correction_iterations = discretization.last_pcg_iterations

    trial_velocity, trial_alpha, _ = discretization.supplementary_update(
        provisional_velocity,
        velocity_previous,
        predicted_midpoint_velocity,
        forcing_load,
        trial_time_step,
    )

    return (
        trial_velocity,
        trial_alpha,
        prediction_iterations,
        correction_iterations,
    )


def parameter_tag(value):
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def save_run_data(
    file_stem,
    data,
    case_index,
    increment_tolerance,
    variation_limit,
    run_elapsed,
    accepted_steps,
    rejected_trials,
):
    npz_file = data_dir / f"{file_stem}.npz"
    csv_file = data_dir / f"{file_stem}.csv"

    np.savez(
        npz_file,
        **data,
        case_index=int(case_index),
        N=N,
        nu=nu,
        final_time=final_time,
        omega=omega,
        amplitude_scale=amplitude_scale,
        window_left=window_left,
        window_right=window_right,
        transition_width=transition_width,
        rho_shift=discretization.rho_shift,
        tau_min=tau_min,
        tau_max=tau_max,
        increment_tolerance=increment_tolerance,
        variation_limit=variation_limit,
        rho_safety=rho_safety,
        r_max=r_max,
        threshold_enabled=True,
        threshold_value=int(threshold),
        run_elapsed=run_elapsed,
        accepted_steps=accepted_steps,
        rejected_trials=rejected_trials,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=discretization.pcg_max_iterations,
    )

    output_matrix = np.column_stack([data[name] for name in column_names])
    formats = (
        ["%d"]
        + ["%.16e"]*8
        + ["%d", "%d", "%d"]
    )
    np.savetxt(
        csv_file,
        output_matrix,
        delimiter=",",
        header=",".join(column_names),
        comments="",
        fmt=formats,
    )
    return npz_file, csv_file


def run_adaptive_experiment(
    case_index,
    increment_tolerance,
    variation_limit,
):
    run_start = timer.perf_counter()
    run_tag = (
        f"case_{case_index:02d}_"
        f"tol_{parameter_tag(increment_tolerance)}_"
        f"var_{parameter_tag(variation_limit)}"
    )
    display_name = (
        f"Tol = {increment_tolerance:.1e}, "
        f"e_limit = {variation_limit:.1e}"
    )

    velocity_previous = discretization.initial_velocity_coefficients()
    velocity_previous_previous = velocity_previous.copy()

    time_previous = 0.0
    accepted_step_number = 1
    previous_time_step = tau_min
    proposed_time_step = tau_min
    counter = 0
    total_rejected_trials = 0
    records = {name: [] for name in column_names}

    time_tolerance = 10.0*np.finfo(float).eps*max(1.0, final_time)

    while time_previous < final_time - time_tolerance:
        if accepted_step_number > maximum_accepted_steps:
            raise RuntimeError("The maximum accepted-step count was exceeded.")

        trial_accepted = False
        rejected_before_acceptance = 0

        while not trial_accepted:
            # Use the complete adaptive step even if it crosses final_time.
            # The last accepted time level is therefore allowed to exceed T,
            # which avoids an artificially small terminal time step.
            trial_time_step = proposed_time_step
            midpoint_time = time_previous + 0.5*trial_time_step

            (
                trial_velocity,
                trial_alpha,
                prediction_iterations,
                correction_iterations,
            ) = compute_pc_cn_trial(
                velocity_previous,
                velocity_previous_previous,
                accepted_step_number,
                previous_time_step,
                trial_time_step,
                midpoint_time,
            )

            velocity_increment = trial_velocity - velocity_previous
            delta = discretization.coefficient_l2_norm(velocity_increment)

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
            if recover_pressure_after_acceptance:
                _pressure_current = discretization.pressure_coefficients(
                    trial_velocity,
                    time_current,
                )

            variation_indicator = delta/trial_time_step
            exact_velocity_l2 = discretization.exact_velocity_l2_norm(
                time_current
            )
            numerical_velocity_l2 = (
                discretization.numerical_velocity_l2_norm(trial_velocity)
            )
            velocity_l2_error, _ = discretization.velocity_errors(
                trial_velocity,
                time_current,
            )

            values = (
                accepted_step_number,
                time_current,
                trial_time_step,
                exact_velocity_l2,
                numerical_velocity_l2,
                velocity_l2_error,
                trial_alpha,
                delta,
                variation_indicator,
                rejected_before_acceptance,
                prediction_iterations,
                correction_iterations,
            )
            for name, value in zip(column_names, values):
                records[name].append(value)

            if variation_indicator < variation_limit:
                counter += 1
                growth_is_allowed = counter >= threshold

                if growth_is_allowed:
                    previous_norm = discretization.coefficient_l2_norm(
                        velocity_previous
                    )
                    trial_norm = discretization.coefficient_l2_norm(
                        trial_velocity
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

            velocity_previous_previous = velocity_previous
            velocity_previous = trial_velocity
            previous_time_step = trial_time_step
            proposed_time_step = next_time_step
            time_previous = time_current
            accepted_step_number += 1

    data = {}
    integer_fields = {
        "step",
        "rejected_trials_before_acceptance",
        "pcg_iterations_prediction",
        "pcg_iterations_correction",
    }
    for name in column_names:
        data[name] = np.asarray(
            records[name],
            dtype=int if name in integer_fields else float,
        )

    file_stem = f"pc_cn_adaptive_multiscale_{run_tag}"
    run_elapsed = timer.perf_counter() - run_start
    accepted_steps = int(data["step"].size)
    npz_file, csv_file = save_run_data(
        file_stem,
        data,
        case_index,
        increment_tolerance,
        variation_limit,
        run_elapsed,
        accepted_steps,
        total_rejected_trials,
    )

    print(f"\nPC-CN adaptive test: {display_name}")
    print(f"  elapsed time = {run_elapsed:.6f} s")
    print(f"  accepted steps = {accepted_steps}")
    print(f"  rejected trials = {total_rejected_trials}")
    print(f"  tau_min used = {np.min(data['tau']):.6e}")
    print(f"  tau_max used = {np.max(data['tau']):.6e}")
    print(
        "  max L2 velocity error = "
        f"{np.max(data['velocity_l2_error']):.6e}"
    )
    print(f"  max |alpha| = {np.max(np.abs(data['alpha'])):.6e}")
    print(f"  NPZ data: {npz_file}")
    print(f"  CSV data: {csv_file}")
    return {
        "case_index": int(case_index),
        "increment_tolerance": float(increment_tolerance),
        "variation_limit": float(variation_limit),
        "run_elapsed": float(run_elapsed),
        "accepted_steps": accepted_steps,
        "rejected_trials": int(total_rejected_trials),
        "npz_file": str(npz_file),
        "csv_file": str(csv_file),
    }


run_summaries = []
for case_index, (increment_tolerance, variation_limit) in enumerate(
    parameter_pairs,
    start=1,
):
    run_summaries.append(
        run_adaptive_experiment(
            case_index,
            increment_tolerance,
            variation_limit,
        )
    )


summary_file = data_dir / "pc_cn_adaptive_multiscale_tolerance_summary.csv"
summary_matrix = np.asarray(
    [
        (
            item["case_index"],
            item["increment_tolerance"],
            item["variation_limit"],
            item["run_elapsed"],
            item["accepted_steps"],
            item["rejected_trials"],
        )
        for item in run_summaries
    ],
    dtype=float,
)
np.savetxt(
    summary_file,
    summary_matrix,
    delimiter=",",
    header=(
        "case_index,increment_tolerance,variation_limit,elapsed_seconds,"
        "accepted_steps,rejected_trials"
    ),
    comments="",
    fmt=("%d", "%.16e", "%.16e", "%.16e", "%d", "%d"),
)

summary_npz_file = (
    data_dir / "pc_cn_adaptive_multiscale_tolerance_summary.npz"
)
np.savez(
    summary_npz_file,
    case_index=summary_matrix[:, 0].astype(int),
    increment_tolerance=summary_matrix[:, 1],
    variation_limit=summary_matrix[:, 2],
    elapsed_seconds=summary_matrix[:, 3],
    accepted_steps=summary_matrix[:, 4].astype(int),
    rejected_trials=summary_matrix[:, 5].astype(int),
)


parameter_file = data_dir / "pc_cn_adaptive_multiscale_parameters.npz"
np.savez(
    parameter_file,
    N=N,
    nu=nu,
    final_time=final_time,
    number_of_windows=number_of_windows,
    omega=omega,
    amplitude_scale=amplitude_scale,
    window_left=window_left,
    window_right=window_right,
    transition_width=transition_width,
    rho_shift=discretization.rho_shift,
    tau_min=tau_min,
    tau_max=tau_max,
    increment_tolerance_values=np.asarray(
        increment_tolerance_values,
        dtype=float,
    ),
    variation_limit_values=np.asarray(
        variation_limit_values,
        dtype=float,
    ),
    rho_safety=rho_safety,
    r_max=r_max,
    threshold_value=int(threshold),
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=discretization.pcg_max_iterations,
)

program_elapsed = timer.perf_counter() - program_start
print(f"\nComputed shift C_* = {discretization.rho_shift:.16e}")
print(f"Parameters saved to: {parameter_file}")
print(f"Summary saved to: {summary_file}")
print(f"Summary NPZ saved to: {summary_npz_file}")
print(f"Total elapsed time: {program_elapsed:.2f} s")
