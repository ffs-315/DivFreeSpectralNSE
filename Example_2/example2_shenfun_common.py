import numpy as np
from pathlib import Path
from numpy.polynomial.legendre import Legendre, leggauss
from scipy.linalg import solve
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import splu
from shenfun import FunctionSpace, TrialFunction, TestFunction, grad, inner


class Example2Discretization:
    def __init__(self, N, nu, quadrature_order=None):
        self.N = int(N)
        self.nu = float(nu)
        self.ndof = self.N - 3

        if self.N < 7:
            raise ValueError("N must satisfy N >= 7")

        if quadrature_order is None:
            quadrature_order = max(3*(self.N + 1), 64)

        self.quadrature_order = int(quadrature_order)
        self.x, self.w = leggauss(self.quadrature_order)
        self.X = self.x[:, None]
        self.Y = self.x[None, :]

        self._build_velocity_basis()
        self._build_pressure_basis()
        self._build_matrices()

    def _build_velocity_basis(self):
        # Shenfun's H_0^2-conforming Legendre space has N-3 unknowns.
        V = FunctionSpace(
            self.N + 1,
            "Legendre",
            bc=(0, 0, 0, 0),
        )
        u = TrialFunction(V)
        v = TestFunction(V)

        M_shen = inner(u, v).diags(format="csr")
        S_shen = inner(grad(u), grad(v)).diags(format="csr")

        k = np.arange(self.ndof, dtype=float)
        scale = 1.0 / np.sqrt(
            2.0*(2.0*k + 3.0)**2*(2.0*k + 5.0)
        )

        D_scale = csr_matrix(
            (
                scale,
                (np.arange(self.ndof), np.arange(self.ndof)),
            ),
            shape=(self.ndof, self.ndof),
        )

        self.M = (D_scale @ M_shen @ D_scale).toarray()
        self.S = (D_scale @ S_shen @ D_scale).toarray()

        # The normalized basis satisfies (phi_i'', phi_j'') = delta_ij.
        self.A = np.eye(self.ndof)

        self.B = np.empty((self.quadrature_order, self.ndof))
        self.G = np.empty_like(self.B)
        self.H = np.empty_like(self.B)

        self.B_end = np.empty((2, self.ndof))
        self.H_end = np.empty((2, self.ndof))

        endpoints = np.array([-1.0, 1.0])

        for j in range(self.ndof):
            coefficient_2 = 2.0*(2.0*j + 5.0)/(2.0*j + 7.0)
            coefficient_4 = (2.0*j + 3.0)/(2.0*j + 7.0)
            normalization = 1.0 / np.sqrt(
                2.0*(2.0*j + 3.0)**2*(2.0*j + 5.0)
            )

            polynomial = normalization*(
                Legendre.basis(j)
                - coefficient_2*Legendre.basis(j + 2)
                + coefficient_4*Legendre.basis(j + 4)
            )

            self.B[:, j] = polynomial(self.x)
            self.G[:, j] = polynomial.deriv(1)(self.x)
            self.H[:, j] = polynomial.deriv(2)(self.x)
            self.B_end[:, j] = polynomial(endpoints)
            self.H_end[:, j] = polynomial.deriv(2)(endpoints)

    def _build_pressure_basis(self):
        # This also checks that the standard pressure space is available in
        # the active Shenfun installation.
        self.pressure_space = FunctionSpace(
            self.N + 1,
            "Legendre",
        )

        pressure_size = self.N + 1
        self.P = np.empty((self.quadrature_order, pressure_size))
        self.Px = np.empty_like(self.P)
        self.P_end = np.empty((2, pressure_size))

        endpoints = np.array([-1.0, 1.0])
        for j in range(pressure_size):
            polynomial = Legendre.basis(j)
            self.P[:, j] = polynomial(self.x)
            self.Px[:, j] = polynomial.deriv(1)(self.x)
            self.P_end[:, j] = polynomial(endpoints)

        weighted_P = self.w[:, None]*self.P
        weighted_Px = self.w[:, None]*self.Px
        self.Mp = self.P.T @ weighted_P
        self.Sp = self.Px.T @ weighted_Px

    def _build_matrices(self):
        M = csr_matrix(self.M)
        S = csr_matrix(self.S)
        A = eye(self.ndof, format="csr")

        self.Mdiv = (
            kron(M, S, format="csr")
            + kron(S, M, format="csr")
        )
        self.Sdiv = (
            kron(A, M, format="csr")
            + 2.0*kron(S, S, format="csr")
            + kron(M, A, format="csr")
        )

        Mp = csr_matrix(self.Mp)
        Sp = csr_matrix(self.Sp)
        pressure_matrix = (
            kron(Sp, Mp, format="csr")
            + kron(Mp, Sp, format="csr")
        )

        # Flat index zero corresponds to L_0(x)L_0(y).
        self.pressure_solver = splu(
            pressure_matrix[1:, 1:].tocsc()
        )

    def nonuniform_grid(self, T, number_of_steps, power=2.0):
        index = np.arange(number_of_steps + 1, dtype=float)
        return float(T)*(index/number_of_steps)**float(power)

    def exact_fields(self, time):
        x = self.X
        y = self.Y
        cosine_time = np.cos(time)

        u1 = (
            cosine_time
            * np.sin(np.pi*x)**2
            * np.sin(2.0*np.pi*y)
        )
        u2 = (
            -cosine_time
            * np.sin(2.0*np.pi*x)
            * np.sin(np.pi*y)**2
        )
        pressure = (
            cosine_time
            * np.cos(np.pi*x)
            * np.sin(np.pi*y)
        )
        return u1, u2, pressure

    def exact_velocity_derivatives(self, time):
        x = self.X
        y = self.Y
        cosine_time = np.cos(time)

        u1_x = (
            cosine_time*np.pi
            * np.sin(2.0*np.pi*x)
            * np.sin(2.0*np.pi*y)
        )
        u1_y = (
            2.0*cosine_time*np.pi
            * np.sin(np.pi*x)**2
            * np.cos(2.0*np.pi*y)
        )
        u2_x = (
            -2.0*cosine_time*np.pi
            * np.cos(2.0*np.pi*x)
            * np.sin(np.pi*y)**2
        )
        u2_y = (
            -cosine_time*np.pi
            * np.sin(2.0*np.pi*x)
            * np.sin(2.0*np.pi*y)
        )
        return u1_x, u1_y, u2_x, u2_y

    def forcing_values(self, time):
        x = self.X
        y = self.Y
        cosine_time = np.cos(time)
        sine_time = np.sin(time)

        spatial_u1 = np.sin(np.pi*x)**2*np.sin(2.0*np.pi*y)
        spatial_u2 = -np.sin(2.0*np.pi*x)*np.sin(np.pi*y)**2

        u1 = cosine_time*spatial_u1
        u2 = cosine_time*spatial_u2

        u1_x, u1_y, u2_x, u2_y = (
            self.exact_velocity_derivatives(time)
        )

        convection_1 = u1*u1_x + u2*u1_y
        convection_2 = u1*u2_x + u2*u2_y

        laplace_u1 = cosine_time*(
            2.0*np.pi**2*np.cos(2.0*np.pi*x)*np.sin(2.0*np.pi*y)
            - 4.0*np.pi**2*np.sin(np.pi*x)**2*np.sin(2.0*np.pi*y)
        )
        laplace_u2 = cosine_time*(
            4.0*np.pi**2*np.sin(2.0*np.pi*x)*np.sin(np.pi*y)**2
            - 2.0*np.pi**2*np.sin(2.0*np.pi*x)*np.cos(2.0*np.pi*y)
        )

        pressure_x = (
            -cosine_time*np.pi
            * np.sin(np.pi*x)*np.sin(np.pi*y)
        )
        pressure_y = (
            cosine_time*np.pi
            * np.cos(np.pi*x)*np.cos(np.pi*y)
        )

        forcing_1 = (
            -sine_time*spatial_u1
            - self.nu*laplace_u1
            + convection_1
            + pressure_x
        )
        forcing_2 = (
            -sine_time*spatial_u2
            - self.nu*laplace_u2
            + convection_2
            + pressure_y
        )
        return forcing_1, forcing_2

    def scalar_projection(self, values):
        weighted_values = (
            self.w[:, None]*values*self.w[None, :]
        )
        right_hand_side = self.B.T @ weighted_values @ self.B
        coefficients = solve(
            self.M,
            right_hand_side,
            assume_a="sym",
        )
        coefficients = solve(
            self.M,
            coefficients.T,
            assume_a="sym",
        ).T
        return coefficients.ravel()

    def initial_velocity_coefficients(self):
        stream_function = (
            np.sin(np.pi*self.X)**2
            * np.sin(np.pi*self.Y)**2
            / np.pi
        )
        return self.scalar_projection(stream_function)

    def vector_load(self, field_1, field_2):
        weighted_1 = (
            self.w[:, None]*field_1*self.w[None, :]
        )
        weighted_2 = (
            self.w[:, None]*field_2*self.w[None, :]
        )
        load_matrix = (
            self.B.T @ weighted_1 @ self.G
            - self.G.T @ weighted_2 @ self.B
        )
        return load_matrix.ravel()

    def forcing_load(self, time):
        forcing_1, forcing_2 = self.forcing_values(time)
        return self.vector_load(forcing_1, forcing_2)

    def velocity_values(self, coefficients):
        coefficient_matrix = coefficients.reshape(
            self.ndof,
            self.ndof,
        )
        velocity_1 = self.B @ coefficient_matrix @ self.G.T
        velocity_2 = -self.G @ coefficient_matrix @ self.B.T
        return velocity_1, velocity_2

    def velocity_derivatives(self, coefficients):
        coefficient_matrix = coefficients.reshape(
            self.ndof,
            self.ndof,
        )
        velocity_1_x = self.G @ coefficient_matrix @ self.G.T
        velocity_1_y = self.B @ coefficient_matrix @ self.H.T
        velocity_2_x = -self.H @ coefficient_matrix @ self.B.T
        velocity_2_y = -self.G @ coefficient_matrix @ self.G.T
        return velocity_1_x, velocity_1_y, velocity_2_x, velocity_2_y

    def convection_load(self, coefficients):
        velocity_1, velocity_2 = self.velocity_values(coefficients)
        velocity_1_x, velocity_1_y, velocity_2_x, velocity_2_y = (
            self.velocity_derivatives(coefficients)
        )

        convection_1 = (
            velocity_1*velocity_1_x
            + velocity_2*velocity_1_y
        )
        convection_2 = (
            velocity_1*velocity_2_x
            + velocity_2*velocity_2_y
        )
        return self.vector_load(convection_1, convection_2)

    def factor_velocity_matrix(self, time_step):
        matrix = (
            self.Mdiv
            + 0.5*self.nu*time_step*self.Sdiv
        )
        return splu(matrix.tocsc())

    def solve_velocity(
        self,
        time_step,
        right_hand_side,
        factorization=None,
    ):
        if factorization is None:
            factorization = self.factor_velocity_matrix(time_step)
        return factorization.solve(right_hand_side)

    def energy(self, coefficients):
        return 0.5*float(coefficients @ (self.Mdiv @ coefficients))

    def gradient_norm_squared(self, coefficients):
        return float(coefficients @ (self.Sdiv @ coefficients))

    def supplementary_update(
        self,
        provisional_coefficients,
        previous_coefficients,
        energy_velocity,
        forcing_load,
        time_step,
    ):
        energy_1 = self.energy(provisional_coefficients)
        energy_previous = self.energy(previous_coefficients)

        dissipation_minus_work = (
            self.nu*self.gradient_norm_squared(energy_velocity)
            - float(energy_velocity @ forcing_load)
        )

        coefficient_a = energy_1
        coefficient_b = 2.0*energy_1
        coefficient_c = (
            energy_1
            - energy_previous
            + time_step*dissipation_minus_work
        )

        discriminant = (
            coefficient_b**2
            - 4.0*coefficient_a*coefficient_c
        )
        discriminant_tolerance = (
            1.0e-12
            * max(coefficient_b**2, 1.0)
        )

        if discriminant < -discriminant_tolerance:
            raise RuntimeError(
                "The supplementary-variable quadratic has a negative "
                f"discriminant: {discriminant:.6e}"
            )

        discriminant = max(discriminant, 0.0)
        square_root = np.sqrt(discriminant)

        denominator = coefficient_b + square_root
        if abs(denominator) < 1.0e-30:
            beta = 0.0
        else:
            beta = -2.0*coefficient_c/denominator

        updated_coefficients = (
            (1.0 + beta)*provisional_coefficients
        )
        alpha = beta/time_step

        energy_residual = (
            (self.energy(updated_coefficients) - energy_previous)
            / time_step
            + dissipation_minus_work
        )

        return updated_coefficients, alpha, energy_residual

    def pressure_coefficients(self, velocity_coefficients, time):
        forcing_1, forcing_2 = self.forcing_values(time)

        velocity_1, velocity_2 = self.velocity_values(
            velocity_coefficients
        )
        velocity_1_x, velocity_1_y, velocity_2_x, velocity_2_y = (
            self.velocity_derivatives(velocity_coefficients)
        )

        convection_1 = velocity_1*velocity_1_x + velocity_2*velocity_1_y
        convection_2 = velocity_1*velocity_2_x + velocity_2*velocity_2_y

        interior_1 = forcing_1 - convection_1
        interior_2 = forcing_2 - convection_2

        weighted_1 = self.w[:, None]*interior_1*self.w[None, :]
        weighted_2 = self.w[:, None]*interior_2*self.w[None, :]

        pressure_load = (
            self.Px.T @ weighted_1 @ self.P
            + self.P.T @ weighted_2 @ self.Px
        )

        coefficient_matrix = velocity_coefficients.reshape(
            self.ndof,
            self.ndof,
        )

        omega_left = -(
            self.H_end[0] @ coefficient_matrix @ self.B.T
            + self.B_end[0] @ coefficient_matrix @ self.H.T
        )
        omega_right = -(
            self.H_end[1] @ coefficient_matrix @ self.B.T
            + self.B_end[1] @ coefficient_matrix @ self.H.T
        )
        omega_bottom = -(
            self.H @ coefficient_matrix @ self.B_end[0]
            + self.B @ coefficient_matrix @ self.H_end[0]
        )
        omega_top = -(
            self.H @ coefficient_matrix @ self.B_end[1]
            + self.B @ coefficient_matrix @ self.H_end[1]
        )

        boundary_load = (
            np.outer(
                self.P_end[1],
                self.Px.T @ (self.w*omega_right),
            )
            - np.outer(
                self.P_end[0],
                self.Px.T @ (self.w*omega_left),
            )
            - np.outer(
                self.Px.T @ (self.w*omega_top),
                self.P_end[1],
            )
            + np.outer(
                self.Px.T @ (self.w*omega_bottom),
                self.P_end[0],
            )
        )

        pressure_load += self.nu*boundary_load

        pressure_flat = np.zeros((self.N + 1)**2)
        pressure_flat[1:] = self.pressure_solver.solve(
            pressure_load.ravel()[1:]
        )
        return pressure_flat

    def velocity_errors(self, coefficients, time):
        numerical_1, numerical_2 = self.velocity_values(coefficients)
        exact_1, exact_2, _ = self.exact_fields(time)

        error_1 = numerical_1 - exact_1
        error_2 = numerical_2 - exact_2
        weighted_error = (
            self.w[:, None]
            * (error_1**2 + error_2**2)
            * self.w[None, :]
        )
        l2_error = np.sqrt(np.sum(weighted_error))

        numerical_derivatives = self.velocity_derivatives(coefficients)
        exact_derivatives = self.exact_velocity_derivatives(time)
        h1_integrand = np.zeros_like(error_1)
        for numerical, exact in zip(
            numerical_derivatives,
            exact_derivatives,
        ):
            h1_integrand += (numerical - exact)**2

        h1_error = np.sqrt(
            np.sum(
                self.w[:, None]
                * h1_integrand
                * self.w[None, :]
            )
        )
        return l2_error, h1_error

    def pressure_error(self, coefficients, time):
        coefficient_matrix = coefficients.reshape(
            self.N + 1,
            self.N + 1,
        )
        numerical_pressure = self.P @ coefficient_matrix @ self.P.T
        _, _, exact_pressure = self.exact_fields(time)
        pressure_difference = numerical_pressure - exact_pressure

        return np.sqrt(
            np.sum(
                self.w[:, None]
                * pressure_difference**2
                * self.w[None, :]
            )
        )

    def divergence_norm(self, coefficients):
        velocity_1_x, _, _, velocity_2_y = (
            self.velocity_derivatives(coefficients)
        )
        divergence = velocity_1_x + velocity_2_y
        return np.sqrt(
            np.sum(
                self.w[:, None]
                * divergence**2
                * self.w[None, :]
            )
        )


def convergence_rates(step_sizes, errors):
    rates = np.full(len(errors), np.nan)
    for index in range(1, len(errors)):
        previous_error = errors[index - 1]
        current_error = errors[index]
        if previous_error > 0.0 and current_error > 0.0:
            rates[index] = (
                np.log(previous_error/current_error)
                / np.log(step_sizes[index - 1]/step_sizes[index])
            )
    return rates


def save_accuracy_data(
    output_file,
    number_of_steps,
    step_sizes,
    velocity_l2,
    pressure_l2,
    divergence,
    alpha_maximum,
):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        number_of_steps=np.asarray(number_of_steps, dtype=int),
        step_sizes=np.asarray(step_sizes, dtype=float),
        velocity_l2=np.asarray(velocity_l2, dtype=float),
        velocity_l2_rate=convergence_rates(step_sizes, velocity_l2),
        pressure_l2=np.asarray(pressure_l2, dtype=float),
        pressure_l2_rate=convergence_rates(step_sizes, pressure_l2),
        divergence_error=np.asarray(divergence, dtype=float),
        divergence_rate=convergence_rates(step_sizes, divergence),
        alpha_error=np.asarray(alpha_maximum, dtype=float),
        alpha_rate=convergence_rates(step_sizes, alpha_maximum),
    )


def print_accuracy_table(output_file, method_name):
    data = np.load(output_file)

    print("\n" + method_name)
    print(
        f"{'M':>5} {'tau_max':>10} "
        f"{'e_u':>11} {'ord':>5} "
        f"{'e_p':>11} {'ord':>5} "
        f"{'e_div':>11} {'ord':>5} "
        f"{'e_alpha':>11} {'ord':>5}"
    )
    print("-" * 88)

    for index in range(len(data["number_of_steps"])):
        def rate_text(key):
            value = data[key][index]
            return "--" if np.isnan(value) else f"{value:.2f}"

        print(
            f"{data['number_of_steps'][index]:5d} "
            f"{data['step_sizes'][index]:10.3e} "
            f"{data['velocity_l2'][index]:11.3e} "
            f"{rate_text('velocity_l2_rate'):>5} "
            f"{data['pressure_l2'][index]:11.3e} "
            f"{rate_text('pressure_l2_rate'):>5} "
            f"{data['divergence_error'][index]:11.3e} "
            f"{rate_text('divergence_rate'):>5} "
            f"{data['alpha_error'][index]:11.3e} "
            f"{rate_text('alpha_rate'):>5}"
        )
