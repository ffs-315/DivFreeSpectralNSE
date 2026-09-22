import numpy as np
from pathlib import Path
import time as timer

program_start = timer.perf_counter()
from example2_shenfun_common_pcg import (
    Example2Discretization,
    print_accuracy_table,
    save_accuracy_data,
)


# ============================================================
# Example 2: parameters for the temporal-accuracy experiment
# ============================================================
N = 80
nu = 1.0e-8
final_time = 1.0
number_of_steps_values = np.array([1000])
r_max = 1.0
random_seed = 20260814
pcg_tolerance = 1.0e-12
pcg_max_iterations = None

data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/example2_accuracy"
)
data_dir.mkdir(parents=True, exist_ok=True)

output_file = data_dir / "example2_imex_cn_accuracy.npz"


discretization = Example2Discretization(
    N=N,
    nu=nu,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=pcg_max_iterations,
)

step_sizes = []
velocity_l2_errors = []
pressure_l2_errors = []
divergence_errors = []
alpha_maxima = []
pcg_iterations_averages = []
pcg_iterations_maxima = []

random_generator = np.random.default_rng(random_seed)


for number_of_steps in number_of_steps_values:
    random_weights = random_generator.uniform(
        1.0/r_max,
        1.0,
        size=int(number_of_steps),
    )
    time_steps = (
        final_time*random_weights/np.sum(random_weights)
    )
    time_grid = np.concatenate(
        (np.array([0.0]), np.cumsum(time_steps))
    )
    time_grid[-1] = final_time

    adjacent_step_ratios = np.maximum(
        time_steps[1:]/time_steps[:-1],
        time_steps[:-1]/time_steps[1:],
    )
    if np.max(adjacent_step_ratios) > r_max*(1.0 + 1.0e-12):
        raise RuntimeError("The random time-step ratio exceeds r_max.")

    velocity_previous = (
        discretization.initial_velocity_coefficients()
    )
    velocity_previous_previous = velocity_previous.copy()

    maximum_divergence = 0.0
    maximum_alpha = 0.0
    pcg_iterations = []
    pcg_relative_residuals = []

    for step in range(1, len(time_grid)):
        time_step = time_steps[step - 1]
        midpoint_time = 0.5*(
            time_grid[step] + time_grid[step - 1]
        )

        if step == 1:
            extrapolated_velocity = velocity_previous.copy()
        else:
            time_step_ratio = (
                time_steps[step - 1]/time_steps[step - 2]
            )
            extrapolated_velocity = (
                (1.0 + 0.5*time_step_ratio)*velocity_previous
                - 0.5*time_step_ratio*velocity_previous_previous
            )

        forcing_load = discretization.forcing_load(midpoint_time)
        convection_load = discretization.convection_load(
            extrapolated_velocity
        )

        right_hand_side = (
            discretization.apply_velocity_mass(velocity_previous)
            - 0.5*nu*time_step
            * discretization.apply_velocity_stiffness(
                velocity_previous
            )
            + time_step*(forcing_load - convection_load)
        )

        provisional_velocity = discretization.solve_velocity(
            time_step,
            right_hand_side,
            initial_guess=velocity_previous,
        )
        pcg_iterations.append(discretization.last_pcg_iterations)
        pcg_relative_residuals.append(
            discretization.last_pcg_relative_residual
        )

        velocity_current, alpha, _energy_residual = (
            discretization.supplementary_update(
                provisional_velocity,
                velocity_previous,
                extrapolated_velocity,
                forcing_load,
                time_step,
            )
        )

        maximum_divergence = max(
            maximum_divergence,
            discretization.divergence_norm(velocity_current),
        )
        maximum_alpha = max(maximum_alpha, abs(alpha))

        velocity_previous_previous = velocity_previous
        velocity_previous = velocity_current

    pressure_current = discretization.pressure_coefficients(
        velocity_previous,
        final_time,
    )

    velocity_l2, _velocity_h1 = discretization.velocity_errors(
        velocity_previous,
        final_time,
    )
    pressure_l2 = discretization.pressure_error(
        pressure_current,
        final_time,
    )

    step_sizes.append(np.max(time_steps))
    velocity_l2_errors.append(velocity_l2)
    pressure_l2_errors.append(pressure_l2)
    divergence_errors.append(maximum_divergence)
    alpha_maxima.append(maximum_alpha)
    pcg_iterations_averages.append(np.mean(pcg_iterations))
    pcg_iterations_maxima.append(np.max(pcg_iterations))

    print(
        f"IMEX-CN: M={number_of_steps:4d}, "
        f"tau_max={np.max(time_steps):.4e}, "
        f"L2(u)={velocity_l2:.4e}, "
        f"L2(p)={pressure_l2:.4e}, "
        f"div_max={maximum_divergence:.4e}, "
        f"alpha_max={maximum_alpha:.4e}, "
        f"PCG(avg/max)={np.mean(pcg_iterations):.1f}/"
        f"{np.max(pcg_iterations):d}, "
        f"res_max={np.max(pcg_relative_residuals):.1e}"
    )


save_accuracy_data(
    output_file,
    number_of_steps_values,
    step_sizes,
    velocity_l2_errors,
    pressure_l2_errors,
    divergence_errors,
    alpha_maxima,
    pcg_iterations_averages,
    pcg_iterations_maxima,
)

np.savez(
    data_dir / "example2_imex_cn_parameters.npz",
    N=N,
    nu=nu,
    final_time=final_time,
    r_max=r_max,
    random_seed=random_seed,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=discretization.pcg_max_iterations,
)

print_accuracy_table(output_file, "Example 2: IMEX-CN scheme (3.6)")
print(f"\nData saved to: {output_file}")

program_elapsed = timer.perf_counter() - program_start
print(f"Total elapsed time: {program_elapsed:.2f} s")