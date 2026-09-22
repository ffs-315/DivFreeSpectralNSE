import time as timer
from pathlib import Path

import numpy as np

from example2_shenfun_common_pcg import Example2Discretization


program_start = timer.perf_counter()


# ============================================================
# PC-CN energy-law experiment
# ============================================================
N = 32
nu = 1.0e-8
final_time = 20.0
number_of_steps = 2000

# Set this to False to use a uniform temporal mesh.
use_random_time_grid = False
r_max = 1.0
random_seed = 20260823

pcg_tolerance = 1.0e-12
pcg_max_iterations = None

data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/test_energy_law"
)
data_dir.mkdir(parents=True, exist_ok=True)


discretization = Example2Discretization(
    N=N,
    nu=nu,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=pcg_max_iterations,
)


# Use exactly the same temporal mesh in the forced and unforced tests.
if use_random_time_grid:
    random_generator = np.random.default_rng(random_seed)
    random_weights = random_generator.uniform(
        1.0/r_max,
        1.0,
        size=number_of_steps,
    )
    time_steps = final_time*random_weights/np.sum(random_weights)
else:
    time_steps = np.full(
        number_of_steps,
        final_time/number_of_steps,
        dtype=float,
    )

time_grid = np.concatenate((np.array([0.0]), np.cumsum(time_steps)))
time_grid[-1] = final_time

if number_of_steps > 1:
    adjacent_step_ratios = np.maximum(
        time_steps[1:]/time_steps[:-1],
        time_steps[:-1]/time_steps[1:],
    )
    if use_random_time_grid and (
        np.max(adjacent_step_ratios) > r_max*(1.0 + 1.0e-12)
    ):
        raise RuntimeError("The random time-step ratio exceeds r_max.")


column_names = (
    "step",
    "time",
    "tau",
    "energy_previous",
    "energy_before",
    "energy_after",
    "dissipation",
    "forcing_work",
    "residual_without_work_before",
    "residual_without_work_after",
    "balance_residual_before",
    "balance_residual_after",
    "cumulative_balance_defect",
    "normalized_cumulative_defect",
    "alpha",
    "beta",
    "pcg_iterations_prediction",
    "pcg_iterations_correction",
    "pcg_relative_residual_prediction",
    "pcg_relative_residual_correction",
)


def save_energy_data(output_stem, data, forcing_enabled):
    npz_file = data_dir / f"{output_stem}.npz"
    csv_file = data_dir / f"{output_stem}.csv"

    np.savez(
        npz_file,
        **data,
        N=N,
        nu=nu,
        final_time=final_time,
        number_of_steps=number_of_steps,
        use_random_time_grid=use_random_time_grid,
        r_max=r_max,
        random_seed=random_seed,
        forcing_enabled=forcing_enabled,
        pcg_tolerance=pcg_tolerance,
        pcg_max_iterations=discretization.pcg_max_iterations,
    )

    output_matrix = np.column_stack([data[name] for name in column_names])
    formats = ["%d"] + ["%.16e"]*15 + ["%d", "%d"] + ["%.16e", "%.16e"]
    np.savetxt(
        csv_file,
        output_matrix,
        delimiter=",",
        header=",".join(column_names),
        comments="",
        fmt=formats,
    )

    return npz_file, csv_file


def run_energy_experiment(forcing_enabled, output_stem):
    velocity_previous = discretization.initial_velocity_coefficients()
    velocity_previous_previous = velocity_previous.copy()
    initial_energy = discretization.energy(velocity_previous)

    records = {name: [] for name in column_names}
    cumulative_balance_defect = 0.0

    for step in range(1, number_of_steps + 1):
        time_step = time_steps[step - 1]
        midpoint_time = 0.5*(time_grid[step] + time_grid[step - 1])

        if step == 1:
            extrapolated_velocity = velocity_previous.copy()
        else:
            time_step_ratio = time_steps[step - 1]/time_steps[step - 2]
            extrapolated_velocity = (
                (1.0 + 0.5*time_step_ratio)*velocity_previous
                - 0.5*time_step_ratio*velocity_previous_previous
            )

        if forcing_enabled:
            forcing_load = discretization.forcing_load(midpoint_time)
        else:
            forcing_load = np.zeros_like(velocity_previous)

        common_right_hand_side = (
            discretization.apply_velocity_mass(velocity_previous)
            - 0.5*nu*time_step
            * discretization.apply_velocity_stiffness(velocity_previous)
        )

        # Stage 1: prediction, formula (3.7).
        predicted_right_hand_side = (
            common_right_hand_side
            + time_step*(
                forcing_load
                - discretization.convection_load(extrapolated_velocity)
            )
        )
        predicted_velocity = discretization.solve_velocity(
            time_step,
            predicted_right_hand_side,
            initial_guess=velocity_previous,
        )
        prediction_iterations = discretization.last_pcg_iterations
        prediction_relative_residual = (
            discretization.last_pcg_relative_residual
        )

        predicted_midpoint_velocity = 0.5*(
            predicted_velocity + velocity_previous
        )

        # Stage 2: correction, formula (3.8).
        corrected_right_hand_side = (
            common_right_hand_side
            + time_step*(
                forcing_load
                - discretization.convection_load(
                    predicted_midpoint_velocity
                )
            )
        )
        provisional_velocity = discretization.solve_velocity(
            time_step,
            corrected_right_hand_side,
            initial_guess=predicted_velocity,
        )
        correction_iterations = discretization.last_pcg_iterations
        correction_relative_residual = (
            discretization.last_pcg_relative_residual
        )

        # provisional_velocity is the velocity before multiplication by
        # (1 + beta_n).  supplementary_update returns
        # velocity_current = (1 + beta_n)*provisional_velocity and
        # alpha_n = beta_n/time_step.
        energy_previous = discretization.energy(velocity_previous)
        energy_before = discretization.energy(provisional_velocity)

        velocity_current, alpha, internal_balance_residual = (
            discretization.supplementary_update(
                provisional_velocity,
                velocity_previous,
                predicted_midpoint_velocity,
                forcing_load,
                time_step,
            )
        )
        beta = time_step*alpha
        energy_after = discretization.energy(velocity_current)

        dissipation = (
            nu
            * discretization.gradient_norm_squared(
                predicted_midpoint_velocity
            )
        )
        forcing_work = float(predicted_midpoint_velocity @ forcing_load)

        residual_without_work_before = (
            (energy_before - energy_previous)/time_step + dissipation
        )
        residual_without_work_after = (
            (energy_after - energy_previous)/time_step + dissipation
        )
        balance_residual_before = (
            residual_without_work_before - forcing_work
        )
        balance_residual_after = (
            residual_without_work_after - forcing_work
        )

        # This check also guards against an accidental mismatch between
        # the recorded energy terms and supplementary_update.
        residual_check_tolerance = 1.0e-10*max(
            1.0,
            abs(internal_balance_residual),
            abs(balance_residual_after),
        )
        if (
            abs(internal_balance_residual - balance_residual_after)
            > residual_check_tolerance
        ):
            raise RuntimeError(
                "The recorded post-correction energy residual is "
                "inconsistent with supplementary_update."
            )

        cumulative_balance_defect += time_step*balance_residual_after
        normalized_cumulative_defect = (
            cumulative_balance_defect/initial_energy
            if initial_energy > 0.0
            else np.nan
        )

        values = (
            step,
            time_grid[step],
            time_step,
            energy_previous,
            energy_before,
            energy_after,
            dissipation,
            forcing_work,
            residual_without_work_before,
            residual_without_work_after,
            balance_residual_before,
            balance_residual_after,
            cumulative_balance_defect,
            normalized_cumulative_defect,
            alpha,
            beta,
            prediction_iterations,
            correction_iterations,
            prediction_relative_residual,
            correction_relative_residual,
        )
        for name, value in zip(column_names, values):
            records[name].append(value)

        velocity_previous_previous = velocity_previous
        velocity_previous = velocity_current

    data = {}
    for name in column_names:
        if name in (
            "step",
            "pcg_iterations_prediction",
            "pcg_iterations_correction",
        ):
            data[name] = np.asarray(records[name], dtype=int)
        else:
            data[name] = np.asarray(records[name], dtype=float)

    npz_file, csv_file = save_energy_data(
        output_stem,
        data,
        forcing_enabled,
    )

    maximum_pre_residual = np.max(np.abs(data["balance_residual_before"]))
    maximum_post_residual = np.max(np.abs(data["balance_residual_after"]))
    maximum_cumulative_defect = np.max(
        np.abs(data["normalized_cumulative_defect"])
    )
    maximum_energy_increase = np.max(
        np.maximum(data["energy_after"] - data["energy_previous"], 0.0)
    )

    experiment_name = "forced" if forcing_enabled else "unforced"
    print(f"\nPC-CN energy-law test ({experiment_name})")
    print(f"  max |R_pre|              = {maximum_pre_residual:.6e}")
    print(f"  max |R_post|             = {maximum_post_residual:.6e}")
    print(f"  max normalized cum. err. = {maximum_cumulative_defect:.6e}")
    if not forcing_enabled:
        print(f"  max positive energy jump = {maximum_energy_increase:.6e}")
    print(f"  NPZ data: {npz_file}")
    print(f"  CSV data: {csv_file}")


run_energy_experiment(
    forcing_enabled=True,
    output_stem="pc_cn_energy_law_forced",
)

run_energy_experiment(
    forcing_enabled=False,
    output_stem="pc_cn_energy_law_unforced",
)


parameter_file = data_dir / "pc_cn_energy_law_parameters.npz"
np.savez(
    parameter_file,
    N=N,
    nu=nu,
    final_time=final_time,
    number_of_steps=number_of_steps,
    use_random_time_grid=use_random_time_grid,
    r_max=r_max,
    random_seed=random_seed,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=discretization.pcg_max_iterations,
    time_grid=time_grid,
    time_steps=time_steps,
)

program_elapsed = timer.perf_counter() - program_start
print(f"\nParameters saved to: {parameter_file}")
print(f"Total elapsed time: {program_elapsed:.2f} s")
