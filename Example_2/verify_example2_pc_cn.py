import numpy as np
from pathlib import Path
import time as timer

program_start = timer.perf_counter()

from example2_shenfun_common import (
    Example2Discretization,
    print_accuracy_table,
    save_accuracy_data,
)


# ============================================================
# Example 2: parameters for the temporal-accuracy experiment
# ============================================================
N = 32
nu = 1.0e-1
final_time = 1.0
number_of_steps_values = np.array([20, 40, 80, 160, 320])
nonuniform_power = 1.0

data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/example2_accuracy"
)
data_dir.mkdir(parents=True, exist_ok=True)

output_file = data_dir / "example2_pc_cn_accuracy.npz"


discretization = Example2Discretization(N=N, nu=nu)

step_sizes = []
velocity_l2_errors = []
pressure_l2_errors = []
divergence_errors = []
alpha_maxima = []


for number_of_steps in number_of_steps_values:
    time = discretization.nonuniform_grid(
        final_time,
        int(number_of_steps),
        power=nonuniform_power,
    )
    time_steps = np.diff(time)

    velocity_previous = (
        discretization.initial_velocity_coefficients()
    )
    velocity_previous_previous = velocity_previous.copy()

    maximum_divergence = 0.0
    maximum_alpha = 0.0

    for step in range(1, len(time)):
        time_step = time_steps[step - 1]
        midpoint_time = 0.5*(time[step] + time[step - 1])

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

        common_right_hand_side = (
            discretization.Mdiv
            - 0.5*nu*time_step*discretization.Sdiv
        ) @ velocity_previous

        # Prediction and correction use the same left-hand-side matrix.
        velocity_factorization = (
            discretization.factor_velocity_matrix(time_step)
        )

        # Stage 1: prediction, formula (3.7).
        predicted_right_hand_side = (
            common_right_hand_side
            + time_step*(
                forcing_load
                - discretization.convection_load(
                    extrapolated_velocity
                )
            )
        )
        predicted_velocity = discretization.solve_velocity(
            time_step,
            predicted_right_hand_side,
            factorization=velocity_factorization,
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
            factorization=velocity_factorization,
        )

        velocity_current, alpha, _energy_residual = (
            discretization.supplementary_update(
                provisional_velocity,
                velocity_previous,
                predicted_midpoint_velocity,
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

    print(
        f"PC-CN: M={number_of_steps:4d}, "
        f"tau_max={np.max(time_steps):.4e}, "
        f"L2(u)={velocity_l2:.4e}, "
        f"L2(p)={pressure_l2:.4e}, "
        f"div_max={maximum_divergence:.4e}, "
        f"alpha_max={maximum_alpha:.4e}"
    )


save_accuracy_data(
    output_file,
    number_of_steps_values,
    step_sizes,
    velocity_l2_errors,
    pressure_l2_errors,
    divergence_errors,
    alpha_maxima,
)

np.savez(
    data_dir / "example2_pc_cn_parameters.npz",
    N=N,
    nu=nu,
    final_time=final_time,
    nonuniform_power=nonuniform_power,
)

print_accuracy_table(output_file, "Example 2: PC-CN schemes (3.7)-(3.8)")
print(f"\nData saved to: {output_file}")


program_elapsed = timer.perf_counter() - program_start
print(f"Total elapsed time: {program_elapsed:.2f} s")