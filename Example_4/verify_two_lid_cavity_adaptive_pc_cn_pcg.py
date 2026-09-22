import time as timer
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from shenfun import FunctionSpace, TrialFunction, TestFunction, grad, inner


program_start = timer.perf_counter()


# ============================================================
# Adaptive PC-CN simulation of a regularized adjacent-wall, two-sided
# cavity on the physical domain (X, Y) in (0, 1)^2.  The upper wall
# moves to the right and the left wall moves downward.
# ============================================================
N = 64
Reynolds_number = 5000.0
nu = 1.0/Reynolds_number
final_time = 1500
lid_speed = 1.0
# Polynomial power used only for extending the prescribed wall velocity
# into the cavity.  It does not change g(s)=16*U_lid*s^2(1-s)^2 on the
# moving walls.  The choice lifting_power=8 gives E(u_L) about 0.01494.
lifting_power = 8

# Adaptive time-step parameters.
tau_min = 5.0e-4
tau_max = 1.0e-2
increment_tolerance = 1.0e-2
variation_limit = 5.0e-2
rho_safety = 0.9
r_max = 2.0
threshold = 50

# Print the accepted step number, t_n, tau_n, and the corrected energy
# every this many accepted steps.  Set it to None to disable progress
# output; for example, use 10, 50, or 100 as needed.
progress_output_interval = 100

pcg_tolerance = 1.0e-12
pcg_max_iterations = None
# A trial step is rejected if its post-correction energy residual exceeds
# this relative tolerance after the stable quadratic solve.
energy_residual_tolerance = 1.0e-11
maximum_accepted_steps = 1_000_000
maximum_rejected_trials = 1_000_000
final_velocity_grid_size = 1000
snapshot_start_time = 1495.0
snapshot_interval_steps = 10
snapshot_velocity_grid_size = final_velocity_grid_size

data_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/"
    "Data/two_lid_problem"
)
data_dir.mkdir(parents=True, exist_ok=True)
snapshot_velocity_dir = data_dir / "velocity_snapshots"
snapshot_velocity_dir.mkdir(parents=True, exist_ok=True)


if N < 7:
    raise ValueError("N must satisfy N >= 7.")
if Reynolds_number <= 0.0:
    raise ValueError("Reynolds_number must be positive.")
if (
    not isinstance(lifting_power, (int, np.integer))
    or lifting_power < 2
):
    raise ValueError("lifting_power must be an integer no smaller than 2.")
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
if final_velocity_grid_size < 2:
    raise ValueError("final_velocity_grid_size must be at least two.")
if snapshot_velocity_grid_size < 2:
    raise ValueError("snapshot_velocity_grid_size must be at least two.")
if snapshot_start_time < 0.0:
    raise ValueError("snapshot_start_time must be nonnegative.")
if snapshot_interval_steps is not None and (
    not isinstance(snapshot_interval_steps, (int, np.integer))
    or snapshot_interval_steps <= 0
):
    raise ValueError(
        "snapshot_interval_steps must be a positive integer or None."
    )
if energy_residual_tolerance <= 0.0:
    raise ValueError("energy_residual_tolerance must be positive.")


class EnergyCorrectionFailure(RuntimeError):
    """Raised when the supplementary-variable equation has no real root."""


class TwoLidCavityDiscretization:
    def __init__(
        self,
        N,
        nu,
        lid_speed=1.0,
        lifting_power=8,
        quadrature_order=None,
        pcg_tolerance=1.0e-12,
        pcg_max_iterations=None,
    ):
        self.N = int(N)
        self.nu = float(nu)
        self.lid_speed = float(lid_speed)
        self.lifting_power = int(lifting_power)
        if self.lifting_power < 2:
            raise ValueError(
                "lifting_power must be an integer no smaller than 2."
            )
        self.ndof = self.N - 3
        self.pcg_tolerance = float(pcg_tolerance)

        if pcg_max_iterations is None:
            pcg_max_iterations = max(200, 8*self.N)
        self.pcg_max_iterations = int(pcg_max_iterations)
        self.last_pcg_iterations = 0
        self.last_pcg_relative_residual = np.nan

        if quadrature_order is None:
            quadrature_order = max(3*(self.N + 1), 64)
        self.quadrature_order = int(quadrature_order)

        # xi and eta are reference coordinates in (-1, 1).  The physical
        # coordinates are X=(xi+1)/2 and Y=(eta+1)/2.
        self.reference_nodes, self.reference_weights = leggauss(
            self.quadrature_order
        )
        self.X = 0.5*(self.reference_nodes[:, None] + 1.0)
        self.Y = 0.5*(self.reference_nodes[None, :] + 1.0)

        self._build_basis_and_matrices()
        self._build_lifting_fields()
        self._build_preconditioner()

    def _build_basis_and_matrices(self):
        V = FunctionSpace(
            self.N + 1,
            "Legendre",
            bc=(0, 0, 0, 0),
        )
        trial = TrialFunction(V)
        test = TestFunction(V)

        mass_shen = inner(trial, test).diags(format="csr")
        stiffness_shen = inner(
            grad(trial),
            grad(test),
        ).diags(format="csr")

        index = np.arange(self.ndof, dtype=float)
        scale = 1.0/np.sqrt(
            2.0*(2.0*index + 3.0)**2*(2.0*index + 5.0)
        )
        scaling = csr_matrix(
            (
                scale,
                (np.arange(self.ndof), np.arange(self.ndof)),
            ),
            shape=(self.ndof, self.ndof),
        )

        self.M_sparse = (scaling @ mass_shen @ scaling).tocsr()
        self.S_sparse = (scaling @ stiffness_shen @ scaling).tocsr()
        self.M = self.M_sparse.toarray()
        self.S = self.S_sparse.toarray()

        self.basis_polynomials = []
        self.B = np.empty((self.quadrature_order, self.ndof))
        self.G = np.empty_like(self.B)
        self.H = np.empty_like(self.B)
        self.G_end = np.empty((2, self.ndof))
        self.H_end = np.empty((2, self.ndof))
        endpoints = np.array([-1.0, 1.0])

        for j in range(self.ndof):
            coefficient_2 = 2.0*(2.0*j + 5.0)/(2.0*j + 7.0)
            coefficient_4 = (2.0*j + 3.0)/(2.0*j + 7.0)
            normalization = 1.0/np.sqrt(
                2.0*(2.0*j + 3.0)**2*(2.0*j + 5.0)
            )
            polynomial = normalization*(
                Legendre.basis(j)
                - coefficient_2*Legendre.basis(j + 2)
                + coefficient_4*Legendre.basis(j + 4)
            )
            self.basis_polynomials.append(polynomial)
            self.B[:, j] = polynomial(self.reference_nodes)
            self.G[:, j] = polynomial.deriv(1)(self.reference_nodes)
            self.H[:, j] = polynomial.deriv(2)(self.reference_nodes)
            self.G_end[:, j] = polynomial.deriv(1)(endpoints)
            self.H_end[:, j] = polynomial.deriv(2)(endpoints)

    def _build_lifting_fields(self):
        X = self.X
        Y = self.Y
        amplitude = self.lid_speed

        # A(s)=s^2(1-s)^2 and its first three derivatives.
        A_X = X**2*(1.0 - X)**2
        A_X_1 = 2.0*X - 6.0*X**2 + 4.0*X**3
        A_X_2 = 2.0 - 12.0*X + 12.0*X**2
        A_X_3 = -12.0 + 24.0*X
        A_Y = Y**2*(1.0 - Y)**2
        A_Y_1 = 2.0*Y - 6.0*Y**2 + 4.0*Y**3
        A_Y_2 = 2.0 - 12.0*Y + 12.0*Y**2
        A_Y_3 = -12.0 + 24.0*Y

        # H_t(Y)=Y^m(Y-1) generates the upper-wall velocity, whereas
        # H_l(X)=X(1-X)^m generates the downward left-wall velocity.
        # Their endpoint derivatives preserve the same wall data for
        # every integer m >= 2.
        m = self.lifting_power
        H_top = Y**m*(Y - 1.0)
        H_top_1 = (m + 1.0)*Y**m - m*Y**(m - 1)
        H_top_2 = (
            m*(m + 1.0)*Y**(m - 1)
            - m*(m - 1.0)*Y**(m - 2)
        )
        H_top_3 = (
            m*(m + 1.0)*(m - 1.0)*Y**(m - 2)
            - m*(m - 1.0)*(m - 2.0)*Y**(m - 3)
        )
        H_left = X*(1.0 - X)**m
        H_left_1 = (
            (1.0 - X)**m
            - m*X*(1.0 - X)**(m - 1)
        )
        H_left_2 = (
            -2.0*m*(1.0 - X)**(m - 1)
            + m*(m - 1.0)*X*(1.0 - X)**(m - 2)
        )
        H_left_3 = (
            3.0*m*(m - 1.0)*(1.0 - X)**(m - 2)
            - m*(m - 1.0)*(m - 2.0)
            * X*(1.0 - X)**(m - 3)
        )

        # The divergence-free lifting is generated by
        # psi_L=16*U_lid*[A(X)H_t(Y)+H_l(X)A(Y)].  It satisfies
        # u(X,1)=(g(X),0), u(0,Y)=(0,-g(Y)), and homogeneous
        # Dirichlet conditions on the lower and right walls, where
        # g(s)=16*U_lid*s^2(1-s)^2.
        self.lift_u1 = 16.0*amplitude*(
            A_X*H_top_1 + H_left*A_Y_1
        )
        self.lift_u2 = -16.0*amplitude*(
            A_X_1*H_top + H_left_1*A_Y
        )

        self.lift_u1_X = 16.0*amplitude*(
            A_X_1*H_top_1 + H_left_1*A_Y_1
        )
        self.lift_u1_Y = 16.0*amplitude*(
            A_X*H_top_2 + H_left*A_Y_2
        )
        self.lift_u2_X = -16.0*amplitude*(
            A_X_2*H_top + H_left_2*A_Y
        )
        self.lift_u2_Y = -16.0*amplitude*(
            A_X_1*H_top_1 + H_left_1*A_Y_1
        )

        lift_laplacian_1 = 16.0*amplitude*(
            A_X_2*H_top_1
            + H_left_2*A_Y_1
            + A_X*H_top_3
            + H_left*A_Y_3
        )
        lift_laplacian_2 = -16.0*amplitude*(
            A_X_3*H_top
            + H_left_3*A_Y
            + A_X_1*H_top_2
            + H_left_1*A_Y_2
        )
        self.lift_viscous_load = self.nu*self.vector_load(
            lift_laplacian_1,
            lift_laplacian_2,
        )

        boundary_coordinate = 0.5*(self.reference_nodes + 1.0)
        boundary_A = boundary_coordinate**2*(1.0 - boundary_coordinate)**2
        boundary_H_top = boundary_coordinate**m*(
            boundary_coordinate - 1.0
        )
        boundary_H_left = boundary_coordinate*(
            1.0 - boundary_coordinate
        )**m

        self.boundary_lid_profile = (
            16.0*self.lid_speed
            * boundary_A
        )
        # Tangential normal derivatives of the lifting on the moving
        # upper and left walls.
        self.boundary_lift_u1_Y_top = 16.0*self.lid_speed*(
            2.0*m*boundary_A + 2.0*boundary_H_left
        )
        self.boundary_lift_u2_X_left = -16.0*self.lid_speed*(
            2.0*boundary_H_top - 2.0*m*boundary_A
        )

    def _build_preconditioner(self):
        eigenvalues, eigenvectors = eigh(
            self.S,
            self.M,
            check_finite=False,
        )
        if np.any(eigenvalues <= 0.0):
            raise RuntimeError(
                "The one-dimensional generalized eigenproblem is not "
                "positive definite."
            )
        self.velocity_eigenvectors = eigenvectors
        self.velocity_eigenvalue_sums = (
            eigenvalues[:, None] + eigenvalues[None, :]
        )

    def apply_velocity_mass(self, coefficients):
        coefficient_matrix = np.asarray(coefficients).reshape(
            self.ndof,
            self.ndof,
        )
        s_v_m = (
            self.M_sparse
            @ (self.S_sparse @ coefficient_matrix).T
        ).T
        m_v_s = (
            self.S_sparse
            @ (self.M_sparse @ coefficient_matrix).T
        ).T
        return (s_v_m + m_v_s).ravel()

    def apply_velocity_stiffness(self, coefficients):
        # Mapping (-1,1)^2 to (0,1)^2 multiplies S^div by four.
        coefficient_matrix = np.asarray(coefficients).reshape(
            self.ndof,
            self.ndof,
        )
        m_v = self.M_sparse @ coefficient_matrix
        v_m = (self.M_sparse @ coefficient_matrix.T).T
        s_v_s = (
            self.S_sparse
            @ (self.S_sparse @ coefficient_matrix).T
        ).T
        reference_result = m_v + 2.0*s_v_s + v_m
        return (4.0*reference_result).ravel()

    def apply_velocity_operator(self, coefficients, gamma):
        return (
            self.apply_velocity_mass(coefficients)
            + float(gamma)*self.apply_velocity_stiffness(coefficients)
        )

    def apply_velocity_preconditioner_inverse(self, residual, gamma):
        residual_matrix = np.asarray(residual).reshape(
            self.ndof,
            self.ndof,
        )
        eigenvectors = self.velocity_eigenvectors
        transformed = eigenvectors.T @ residual_matrix @ eigenvectors
        eigenvalue_sums = self.velocity_eigenvalue_sums
        denominator = (
            eigenvalue_sums
            + 4.0*float(gamma)*eigenvalue_sums**2
        )
        solution_matrix = (
            eigenvectors
            @ (transformed/denominator)
            @ eigenvectors.T
        )
        return solution_matrix.ravel()

    def solve_velocity(self, time_step, right_hand_side, initial_guess=None):
        gamma = 0.5*self.nu*float(time_step)
        right_hand_side = np.asarray(right_hand_side, dtype=float)
        if initial_guess is None:
            solution = np.zeros_like(right_hand_side)
        else:
            solution = np.asarray(initial_guess, dtype=float).copy()

        right_hand_side_norm = np.linalg.norm(right_hand_side)
        if right_hand_side_norm == 0.0:
            self.last_pcg_iterations = 0
            self.last_pcg_relative_residual = 0.0
            return np.zeros_like(right_hand_side)

        residual = (
            right_hand_side
            - self.apply_velocity_operator(solution, gamma)
        )
        tolerance = self.pcg_tolerance*right_hand_side_norm
        residual_norm = np.linalg.norm(residual)
        if residual_norm <= tolerance:
            self.last_pcg_iterations = 0
            self.last_pcg_relative_residual = (
                residual_norm/right_hand_side_norm
            )
            return solution

        preconditioned_residual = (
            self.apply_velocity_preconditioner_inverse(residual, gamma)
        )
        search_direction = preconditioned_residual.copy()
        residual_inner_product = float(
            residual @ preconditioned_residual
        )
        if residual_inner_product <= 0.0:
            raise RuntimeError("The PCG preconditioner is not positive definite.")

        for iteration in range(1, self.pcg_max_iterations + 1):
            operator_direction = self.apply_velocity_operator(
                search_direction,
                gamma,
            )
            denominator = float(search_direction @ operator_direction)
            if denominator <= 0.0:
                raise RuntimeError("The velocity operator is not positive definite.")

            step_length = residual_inner_product/denominator
            solution += step_length*search_direction
            residual -= step_length*operator_direction
            residual_norm = np.linalg.norm(residual)

            if residual_norm <= tolerance:
                self.last_pcg_iterations = iteration
                self.last_pcg_relative_residual = (
                    residual_norm/right_hand_side_norm
                )
                return solution

            new_preconditioned_residual = (
                self.apply_velocity_preconditioner_inverse(residual, gamma)
            )
            new_inner_product = float(
                residual @ new_preconditioned_residual
            )
            if new_inner_product <= 0.0:
                raise RuntimeError("PCG lost positive definiteness.")
            search_direction = (
                new_preconditioned_residual
                + (new_inner_product/residual_inner_product)
                * search_direction
            )
            residual_inner_product = new_inner_product

        self.last_pcg_iterations = self.pcg_max_iterations
        self.last_pcg_relative_residual = (
            residual_norm/right_hand_side_norm
        )
        raise RuntimeError(
            "Matrix-free PCG did not converge in "
            f"{self.pcg_max_iterations} iterations; relative residual = "
            f"{self.last_pcg_relative_residual:.3e}."
        )

    def vector_load(self, field_1, field_2):
        weighted_1 = (
            self.reference_weights[:, None]
            * field_1
            * self.reference_weights[None, :]
        )
        weighted_2 = (
            self.reference_weights[:, None]
            * field_2
            * self.reference_weights[None, :]
        )
        # A physical velocity test function contributes a factor two,
        # while dX dY contributes the Jacobian 1/4.
        load_matrix = 0.5*(
            self.B.T @ weighted_1 @ self.G
            - self.G.T @ weighted_2 @ self.B
        )
        return load_matrix.ravel()

    def homogeneous_velocity_values(self, coefficients):
        coefficient_matrix = np.asarray(coefficients).reshape(
            self.ndof,
            self.ndof,
        )
        velocity_1 = 2.0*self.B @ coefficient_matrix @ self.G.T
        velocity_2 = -2.0*self.G @ coefficient_matrix @ self.B.T
        return velocity_1, velocity_2

    def homogeneous_velocity_derivatives(self, coefficients):
        coefficient_matrix = np.asarray(coefficients).reshape(
            self.ndof,
            self.ndof,
        )
        velocity_1_X = 4.0*self.G @ coefficient_matrix @ self.G.T
        velocity_1_Y = 4.0*self.B @ coefficient_matrix @ self.H.T
        velocity_2_X = -4.0*self.H @ coefficient_matrix @ self.B.T
        velocity_2_Y = -4.0*self.G @ coefficient_matrix @ self.G.T
        return velocity_1_X, velocity_1_Y, velocity_2_X, velocity_2_Y

    def total_velocity_values(self, homogeneous_coefficients):
        homogeneous_1, homogeneous_2 = self.homogeneous_velocity_values(
            homogeneous_coefficients
        )
        return (
            self.lift_u1 + homogeneous_1,
            self.lift_u2 + homogeneous_2,
        )

    def total_velocity_derivatives(self, homogeneous_coefficients):
        derivatives = self.homogeneous_velocity_derivatives(
            homogeneous_coefficients
        )
        return (
            self.lift_u1_X + derivatives[0],
            self.lift_u1_Y + derivatives[1],
            self.lift_u2_X + derivatives[2],
            self.lift_u2_Y + derivatives[3],
        )

    def convection_load(self, homogeneous_coefficients):
        velocity_1, velocity_2 = self.total_velocity_values(
            homogeneous_coefficients
        )
        velocity_1_X, velocity_1_Y, velocity_2_X, velocity_2_Y = (
            self.total_velocity_derivatives(homogeneous_coefficients)
        )
        convection_1 = (
            velocity_1*velocity_1_X + velocity_2*velocity_1_Y
        )
        convection_2 = (
            velocity_1*velocity_2_X + velocity_2*velocity_2_Y
        )
        return self.vector_load(convection_1, convection_2)

    def physical_integral(self, values):
        return 0.25*float(
            np.sum(
                self.reference_weights[:, None]
                * values
                * self.reference_weights[None, :]
            )
        )

    def homogeneous_energy(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        return 0.5*float(
            coefficients @ self.apply_velocity_mass(coefficients)
        )

    def total_energy(self, homogeneous_coefficients):
        velocity_1, velocity_2 = self.total_velocity_values(
            homogeneous_coefficients
        )
        return 0.5*self.physical_integral(
            velocity_1**2 + velocity_2**2
        )

    def lift_homogeneous_inner_product(self, coefficients):
        homogeneous_1, homogeneous_2 = self.homogeneous_velocity_values(
            coefficients
        )
        return self.physical_integral(
            self.lift_u1*homogeneous_1
            + self.lift_u2*homogeneous_2
        )

    def total_gradient_norm_squared(self, homogeneous_coefficients):
        derivatives = self.total_velocity_derivatives(
            homogeneous_coefficients
        )
        integrand = np.zeros_like(derivatives[0])
        for derivative in derivatives:
            integrand += derivative**2
        return self.physical_integral(integrand)

    def wall_power(self, homogeneous_coefficients):
        coefficient_matrix = np.asarray(homogeneous_coefficients).reshape(
            self.ndof,
            self.ndof,
        )
        homogeneous_u1_Y_top = (
            4.0*self.B @ coefficient_matrix @ self.H_end[1]
        )
        homogeneous_u2_X_left = (
            -4.0*self.H_end[0] @ coefficient_matrix @ self.B.T
        )
        total_u1_Y_top = (
            self.boundary_lift_u1_Y_top + homogeneous_u1_Y_top
        )
        total_u2_X_left = (
            self.boundary_lift_u2_X_left + homogeneous_u2_X_left
        )
        # The upper wall contributes g(X)*d_Y u_1.  On the left wall,
        # n=(-1,0) and u=(0,-g(Y)); hence u dot d_n u contributes
        # g(Y)*d_X u_2.
        boundary_integral = 0.5*float(
            np.sum(
                self.reference_weights
                * self.boundary_lid_profile
                * (total_u1_Y_top + total_u2_X_left)
            )
        )
        return self.nu*boundary_integral

    def divergence_norm(self, homogeneous_coefficients):
        velocity_1_X, _, _, velocity_2_Y = (
            self.total_velocity_derivatives(homogeneous_coefficients)
        )
        divergence = velocity_1_X + velocity_2_Y
        return np.sqrt(max(self.physical_integral(divergence**2), 0.0))

    def homogeneous_l2_norm(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        squared_norm = float(
            coefficients @ self.apply_velocity_mass(coefficients)
        )
        return np.sqrt(max(squared_norm, 0.0))

    def supplementary_update(
        self,
        provisional_coefficients,
        previous_coefficients,
        energy_coefficients,
        time_step,
    ):
        energy_previous = self.total_energy(previous_coefficients)
        energy_before = self.total_energy(provisional_coefficients)
        dissipation = (
            self.nu
            * self.total_gradient_norm_squared(energy_coefficients)
        )
        wall_work = self.wall_power(energy_coefficients)
        balance_term = dissipation - wall_work

        coefficient_a = self.homogeneous_energy(
            provisional_coefficients
        )
        coefficient_b = (
            self.lift_homogeneous_inner_product(
                provisional_coefficients
            )
            + 2.0*coefficient_a
        )
        coefficient_c = (
            energy_before
            - energy_previous
            + time_step*balance_term
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
                f"discriminant={discriminant:.6e}, "
                "relative discriminant="
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
                # This branch covers a repeated root at, or extremely
                # close to, zero.
                beta = -coefficient_b/(2.0*coefficient_a)
            else:
                # The two roots q/A and C/q are algebraically equivalent
                # to the quadratic formula but avoid subtracting two
                # nearly equal numbers.  The root closest to zero gives
                # the smallest supplementary-variable correction.
                roots = np.array(
                    [
                        q/coefficient_a,
                        coefficient_c/q,
                    ]
                )
                beta = float(roots[np.argmin(np.abs(roots))])

        corrected_coefficients = (
            (1.0 + beta)*np.asarray(provisional_coefficients)
        )
        alpha = beta/time_step
        energy_after = self.total_energy(corrected_coefficients)
        residual_pre = coefficient_c/time_step
        residual_post = (
            (energy_after - energy_previous)/time_step + balance_term
        )
        residual_scale = max(
            abs((energy_after - energy_previous)/time_step),
            abs(dissipation),
            abs(wall_work),
            1.0,
        )
        if (
            not np.isfinite(residual_post)
            or abs(residual_post)
            > energy_residual_tolerance*residual_scale
        ):
            raise EnergyCorrectionFailure(
                "The supplementary-variable correction did not satisfy "
                "the discrete energy equation: "
                f"R_post={residual_post:.6e}, "
                "scaled residual="
                f"{abs(residual_post)/residual_scale:.6e}."
            )

        return {
            "corrected_coefficients": corrected_coefficients,
            "alpha": alpha,
            "energy_previous": energy_previous,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "dissipation": dissipation,
            "wall_work": wall_work,
            "residual_pre": residual_pre,
            "residual_post": residual_post,
        }

    def evaluate_final_velocity(self, coefficients, grid_size):
        physical_grid = np.linspace(0.0, 1.0, int(grid_size))
        reference_grid = 2.0*physical_grid - 1.0
        basis = np.empty((physical_grid.size, self.ndof))
        derivative = np.empty_like(basis)
        for j, polynomial in enumerate(self.basis_polynomials):
            basis[:, j] = polynomial(reference_grid)
            derivative[:, j] = polynomial.deriv(1)(reference_grid)

        coefficient_matrix = np.asarray(coefficients).reshape(
            self.ndof,
            self.ndof,
        )
        # Matrix products are formed as [X,Y] and transposed to [Y,X]
        # for Matplotlib's streamplot convention.
        homogeneous_1 = (
            2.0*basis @ coefficient_matrix @ derivative.T
        ).T
        homogeneous_2 = (
            -2.0*derivative @ coefficient_matrix @ basis.T
        ).T

        X = physical_grid[None, :]
        Y = physical_grid[:, None]
        A_X = X**2*(1.0 - X)**2
        A_X_1 = 2.0*X - 6.0*X**2 + 4.0*X**3
        A_Y = Y**2*(1.0 - Y)**2
        A_Y_1 = 2.0*Y - 6.0*Y**2 + 4.0*Y**3
        m = self.lifting_power
        H_top = Y**m*(Y - 1.0)
        H_top_1 = (m + 1.0)*Y**m - m*Y**(m - 1)
        H_left = X*(1.0 - X)**m
        H_left_1 = (
            (1.0 - X)**m
            - m*X*(1.0 - X)**(m - 1)
        )
        lift_1 = 16.0*self.lid_speed*(
            A_X*H_top_1 + H_left*A_Y_1
        )
        lift_2 = -16.0*self.lid_speed*(
            A_X_1*H_top + H_left_1*A_Y
        )
        return (
            physical_grid,
            physical_grid.copy(),
            lift_1 + homogeneous_1,
            lift_2 + homogeneous_2,
        )


discretization = TwoLidCavityDiscretization(
    N=N,
    nu=nu,
    lid_speed=lid_speed,
    lifting_power=lifting_power,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=pcg_max_iterations,
)


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
    "wall_work",
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
):
    if accepted_step_number == 1:
        extrapolated_velocity = velocity_previous.copy()
    else:
        time_step_ratio = trial_time_step/previous_time_step
        extrapolated_velocity = (
            (1.0 + 0.5*time_step_ratio)*velocity_previous
            - 0.5*time_step_ratio*velocity_previous_previous
        )

    common_right_hand_side = (
        discretization.apply_velocity_mass(velocity_previous)
        - 0.5*nu*trial_time_step
        * discretization.apply_velocity_stiffness(velocity_previous)
    )
    physical_load = discretization.lift_viscous_load

    predicted_right_hand_side = (
        common_right_hand_side
        + trial_time_step*(
            physical_load
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
            physical_load
            - discretization.convection_load(
                predicted_midpoint_velocity
            )
        )
    )
    provisional_velocity = discretization.solve_velocity(
        trial_time_step,
        corrected_right_hand_side,
        initial_guess=predicted_velocity,
    )
    correction_iterations = discretization.last_pcg_iterations

    update = discretization.supplementary_update(
        provisional_velocity,
        velocity_previous,
        predicted_midpoint_velocity,
        trial_time_step,
    )
    update["provisional_coefficients"] = provisional_velocity
    update["prediction_iterations"] = prediction_iterations
    update["correction_iterations"] = correction_iterations
    return update


velocity_previous = np.zeros(discretization.ndof**2)
velocity_previous_previous = velocity_previous.copy()
time_previous = 0.0
accepted_step_number = 1
previous_time_step = tau_min
proposed_time_step = tau_min
counter = 0
total_rejected_trials = 0
records = {name: [] for name in history_columns}
time_tolerance = 10.0*np.finfo(float).eps*max(1.0, final_time)
snapshot_reference_step = None
saved_velocity_snapshot_files = []


def save_velocity_snapshot(coefficients, current_time, step_number):
    grid_x, grid_y, velocity_x, velocity_y = (
        discretization.evaluate_final_velocity(
            coefficients,
            snapshot_velocity_grid_size,
        )
    )
    snapshot_file = (
        snapshot_velocity_dir
        / (
            f"two_lid_pc_cn_velocity_t_{current_time:.2f}_"
            f"{snapshot_velocity_grid_size}x{snapshot_velocity_grid_size}.npz"
        )
    )
    np.savez_compressed(
        snapshot_file,
        x=grid_x,
        y=grid_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        homogeneous_streamfunction_coefficients=coefficients,
        N=N,
        Reynolds_number=Reynolds_number,
        nu=nu,
        time=current_time,
        step=step_number,
        lid_speed=lid_speed,
        lifting_power=lifting_power,
    )
    saved_velocity_snapshot_files.append(snapshot_file)
    return snapshot_file


while time_previous < final_time - time_tolerance:
    if accepted_step_number > maximum_accepted_steps:
        raise RuntimeError("The maximum accepted-step count was exceeded.")

    trial_accepted = False
    rejected_before_acceptance = 0

    while not trial_accepted:
        # The last accepted step may pass T.  This avoids an artificial,
        # extremely small terminal step.
        trial_time_step = proposed_time_step

        try:
            trial = compute_pc_cn_trial(
                velocity_previous,
                velocity_previous_previous,
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

        trial_velocity = trial["corrected_coefficients"]
        delta = discretization.homogeneous_l2_norm(
            trial_velocity - velocity_previous
        )
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
        provisional_velocity = trial["provisional_coefficients"]

        values = (
            accepted_step_number,
            time_current,
            trial_time_step,
            trial["energy_before"],
            trial["energy_after"],
            discretization.divergence_norm(provisional_velocity),
            discretization.divergence_norm(trial_velocity),
            trial["alpha"],
            trial["residual_pre"],
            trial["residual_post"],
            trial["dissipation"],
            trial["wall_work"],
            delta,
            variation_indicator,
            rejected_before_acceptance,
            trial["prediction_iterations"],
            trial["correction_iterations"],
        )
        for name, value in zip(history_columns, values):
            records[name].append(value)

        if (
            progress_output_interval is not None
            and accepted_step_number % progress_output_interval == 0
        ):
            print(
                f"step={accepted_step_number:7d}, "
                f"t_n={time_current:.2f}, "
                f"tau_n={trial_time_step:.2e}, "
                f"e_n={variation_indicator:.2e}, "
                f"E(u^n)={trial['energy_after']:.4e}"
            )

        if variation_indicator < variation_limit:
            counter += 1
            if counter >= threshold:
                previous_norm = discretization.homogeneous_l2_norm(
                    velocity_previous
                )
                trial_norm = discretization.homogeneous_l2_norm(
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

        if (
            snapshot_interval_steps is not None
            and time_current > snapshot_start_time
        ):
            if snapshot_reference_step is None:
                snapshot_reference_step = accepted_step_number
            if (
                accepted_step_number - snapshot_reference_step
            ) % snapshot_interval_steps == 0:
                save_velocity_snapshot(
                    velocity_previous,
                    time_current,
                    accepted_step_number,
                )

        accepted_step_number += 1


integer_fields = {
    "step",
    "rejected_trials_before_acceptance",
    "pcg_iterations_prediction",
    "pcg_iterations_correction",
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


history_npz_file = data_dir / "two_lid_pc_cn_adaptive_history.npz"
history_csv_file = data_dir / "two_lid_pc_cn_adaptive_history.csv"
np.savez(
    history_npz_file,
    **history,
    N=N,
    Reynolds_number=Reynolds_number,
    nu=nu,
    requested_final_time=final_time,
    attained_final_time=history["time"][-1],
    lid_speed=lid_speed,
    lifting_power=lifting_power,
    tau_min=tau_min,
    tau_max=tau_max,
    increment_tolerance=increment_tolerance,
    variation_limit=variation_limit,
    rho_safety=rho_safety,
    r_max=r_max,
    threshold=threshold,
    progress_output_interval=(
        -1
        if progress_output_interval is None
        else int(progress_output_interval)
    ),
    accepted_steps=accepted_steps,
    rejected_trials=total_rejected_trials,
    elapsed_seconds=run_elapsed,
    pcg_tolerance=pcg_tolerance,
    pcg_max_iterations=discretization.pcg_max_iterations,
    energy_residual_tolerance=energy_residual_tolerance,
    initial_condition="u_L (zero homogeneous coefficients)",
    boundary_configuration=(
        "upper wall moves right; left wall moves downward; "
        "lower and right walls are stationary"
    ),
)

history_matrix = np.column_stack(
    [history[name] for name in history_columns]
)
history_formats = (
    ["%d"]
    + ["%.16e"]*13
    + ["%d", "%d", "%d"]
)
np.savetxt(
    history_csv_file,
    history_matrix,
    delimiter=",",
    header=",".join(history_columns),
    comments="",
    fmt=history_formats,
)


grid_x, grid_y, final_u, final_v = (
    discretization.evaluate_final_velocity(
        velocity_previous,
        final_velocity_grid_size,
    )
)
final_velocity_file = (
    data_dir / "two_lid_pc_cn_final_velocity_1000x1000.npz"
)
np.savez_compressed(
    final_velocity_file,
    x=grid_x,
    y=grid_y,
    velocity_x=final_u,
    velocity_y=final_v,
    homogeneous_streamfunction_coefficients=velocity_previous,
    N=N,
    Reynolds_number=Reynolds_number,
    nu=nu,
    time=history["time"][-1],
    lid_speed=lid_speed,
    lifting_power=lifting_power,
)


print("\nAdaptive PC-CN adjacent-wall two-sided cavity simulation")
print(f"  N = {N}")
print(f"  Reynolds number = {Reynolds_number:.6g}")
print(f"  lifting power = {lifting_power}")
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
print(f"  final velocity: {final_velocity_file}")
print(f"  velocity snapshots saved: {len(saved_velocity_snapshot_files)}")
print(f"  velocity snapshot directory: {snapshot_velocity_dir}")
