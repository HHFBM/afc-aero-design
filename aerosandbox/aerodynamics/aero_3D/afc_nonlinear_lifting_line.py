from aerosandbox.aerodynamics.aero_3D.afc_lifting_line import (
    AFCNeuralFoilLiftingLine,
)
from aerosandbox.aerodynamics.aero_3D.nonlinear_lifting_line import (
    NonlinearLiftingLine,
    tall,
    wide,
)
from aerosandbox.aerodynamics.aero_3D.afc_postprocessing import (
    add_afc_3d_postprocessing,
)
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
from typing import Callable, Dict, List, Union


class AFCNeuralFoilNonlinearLiftingLine(NonlinearLiftingLine):
    """
    Nonlinear finite-wing AFC analysis with an implicit circulation solve.

    This analysis couples the 3D lifting-line circulation `gamma` to the local 2D AFC-NeuralFoil section model through
    the residual

        R_i = CL_from_gamma_i - CL_2D(alpha_eff_i, Re_i, Mach_i, C_mu_i)

    The first implementation only enforces local `CL` consistency. Section `CM` is reported for diagnostics, but is
    not added as an extra section pitching-moment/cambering correction in the aggregate 3D moment.
    """

    def __init__(
        self,
        airplane: Airplane,
        op_point: OperatingPoint,
        xyz_ref: List[float] = None,
        C_mu: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.0,
        x_jet: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.1,
        theta_jet: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 30.0,
        model_size: str = "medium",
        weights_directory: str = None,
        training_statistics_path: str = None,
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 4,
        spanwise_spacing_function: Callable[
            [float, float, float], np.ndarray
        ] = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        include_360_deg_effects: bool = True,
        residual_tolerance_CL: float = 1e-6,
        opti=None,
    ):
        super().__init__(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            run_symmetric_if_possible=run_symmetric_if_possible,
            verbose=verbose,
            spanwise_resolution=spanwise_resolution,
            spanwise_spacing_function=spanwise_spacing_function,
            vortex_core_radius=vortex_core_radius,
            align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
            opti=opti,
        )
        self.C_mu = C_mu
        self.x_jet = x_jet
        self.theta_jet = theta_jet
        self.model_size = model_size
        self.weights_directory = weights_directory
        self.training_statistics_path = training_statistics_path
        self.include_360_deg_effects = include_360_deg_effects
        self.residual_tolerance_CL = residual_tolerance_CL

    def run(
        self, solve: bool = True
    ) -> Dict[str, Union[float, np.ndarray, List[Union[float, np.ndarray]]]]:
        """
        Computes AFC nonlinear lifting-line aerodynamic coefficients and spanwise diagnostics.

        Args:
            solve: If True, internally constrains the circulation residuals to zero and solves this analysis. If False,
                returns symbolic residuals and the `gamma` variable so an external `asb.Opti` problem can constrain them.

        Returns:
            A dictionary containing aggregate aerodynamic outputs, the implicit `gamma`, `residuals`, and spanwise
            arrays including `local_alpha`, `local_Re`, `local_C_mu`, `section_CL_from_gamma`, and `section_CL`.
        """
        self.solve = solve

        if self.verbose:
            print("Meshing...")

        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        airfoils: List[Airfoil] = []
        control_surfaces: List[List[ControlSurface]] = []

        for wing in self.airplane.wings:
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=1,
                add_camber=False,
            )

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])

            wing_airfoils = []
            wing_control_surfaces = []
            for xsec_a, xsec_b in zip(wing.xsecs[:-1], wing.xsecs[1:]):
                wing_airfoils.append(
                    xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil,
                        blend_fraction=0.5,
                    )
                )
                wing_control_surfaces.append(xsec_a.control_surfaces)

            airfoils.extend(wing_airfoils)
            control_surfaces.extend(wing_control_surfaces)

            if wing.symmetric:
                airfoils.extend(wing_airfoils)

                def mirror_control_surface(surf: ControlSurface) -> ControlSurface:
                    if surf.symmetric:
                        return surf
                    else:
                        surf = surf.copy()
                        surf.deflection *= -1
                        return surf

                control_surfaces.extend(
                    [
                        [mirror_control_surface(surf) for surf in surfs]
                        for surfs in wing_control_surfaces
                    ]
                )

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)

        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2

        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        vortex_bound_leg_norm = np.linalg.norm(vortex_bound_leg, axis=1)
        chord_vectors = (back_left_vertices + back_right_vertices) / 2 - (
            front_left_vertices + front_right_vertices
        ) / 2
        chords = np.linalg.norm(chord_vectors, axis=1)
        wing_directions = vortex_bound_leg / tall(vortex_bound_leg_norm)
        local_forward_direction = np.cross(normal_directions, wing_directions)

        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.airfoils = airfoils
        self.control_surfaces = control_surfaces
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.chord_vectors = chord_vectors
        self.chords = chords
        self.local_forward_direction = local_forward_direction
        self.n_panels = areas.shape[0]

        spanwise_y = vortex_centers[:, 1]
        local_C_mu = AFCNeuralFoilLiftingLine._spanwise_distribution(
            self.C_mu, spanwise_y, "C_mu"
        )
        local_x_jet = AFCNeuralFoilLiftingLine._spanwise_distribution(
            self.x_jet, spanwise_y, "x_jet"
        )
        local_theta_jet = AFCNeuralFoilLiftingLine._spanwise_distribution(
            self.theta_jet, spanwise_y, "theta_jet"
        )

        if self.verbose:
            print("Calculating freestream velocities...")
        steady_freestream_velocity = (
            self.op_point.compute_freestream_velocity_geometry_axes()
        )
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(
            steady_freestream_velocity
        )
        rotation_freestream_velocities = (
            self.op_point.compute_rotation_velocity_geometry_axes(vortex_centers)
        )
        freestream_velocities = (
            np.tile(wide(steady_freestream_velocity), reps=(self.n_panels, 1))
            + rotation_freestream_velocities
        )

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        if self.verbose:
            print("Creating implicit vortex strengths...")
        vortex_strengths = self.opti.variable(init_guess=np.zeros(shape=self.n_panels))
        self.vortex_strengths = vortex_strengths
        self.gamma = vortex_strengths

        velocities_for_residual = self.get_velocity_at_points(
            points=vortex_centers,
            vortex_strengths=vortex_strengths,
        )
        velocity_magnitudes_for_residual = np.linalg.norm(velocities_for_residual, axis=1)
        velocity_directions = velocities_for_residual / tall(
            velocity_magnitudes_for_residual
        )

        local_alphas = 90 - np.arccosd(
            np.sum(velocity_directions * normal_directions, axis=1)
        )
        cos_sweeps = np.sum(velocity_directions * -local_forward_direction, axis=1)
        velocity_magnitude_perpendiculars = (
            velocity_magnitudes_for_residual * cos_sweeps
        )
        local_Res = (
            velocity_magnitudes_for_residual
            * chords
            / self.op_point.atmosphere.kinematic_viscosity()
        ) * cos_sweeps
        local_machs = (
            velocity_magnitudes_for_residual
            / self.op_point.atmosphere.speed_of_sound()
            * cos_sweeps
        )

        section_aeros = [
            af.get_aero_from_afc_neuralfoil(
                alpha=local_alphas[i],
                Re=local_Res[i],
                mach=local_machs[i],
                C_mu=local_C_mu[i],
                x_jet=local_x_jet[i],
                theta_jet=local_theta_jet[i],
                control_surfaces=control_surfaces[i],
                model_size=self.model_size,
                weights_directory=self.weights_directory,
                training_statistics_path=self.training_statistics_path,
                include_360_deg_effects=self.include_360_deg_effects,
            )
            for i, af in enumerate(airfoils)
        ]
        CL_2D = np.reshape(np.array([aero["CL"] for aero in section_aeros]), -1)
        CD_2D = np.reshape(np.array([aero["CD"] for aero in section_aeros]), -1)
        CM_2D = np.reshape(np.array([aero["CM"] for aero in section_aeros]), -1)
        analysis_confidences = np.reshape(
            np.array([aero["analysis_confidence"] for aero in section_aeros]), -1
        )

        Vi_cross_li_for_residual = np.cross(
            velocities_for_residual, vortex_bound_leg, axis=1
        )
        Vi_cross_li_magnitudes = np.linalg.norm(Vi_cross_li_for_residual, axis=1)
        CL_from_gamma = (
            vortex_strengths
            * Vi_cross_li_magnitudes
            * 2
            / velocity_magnitude_perpendiculars**2
            / areas
        )
        residuals = CL_from_gamma - CL_2D

        if solve:
            if self.verbose:
                print("Solving nonlinear circulation residuals...")
            self.opti.subject_to(residuals == 0)
            try:
                self.sol = self.opti.solve(verbose=False)
                failure_reason = None
            except Exception as e:
                self.failure_reason = f"{type(e).__name__}: {e}"
                raise
            vortex_strengths = self.sol(vortex_strengths)
            self.vortex_strengths = vortex_strengths
            self.gamma = vortex_strengths
        else:
            failure_reason = "solve_false_external_residuals_not_constrained"

        velocities = self.get_velocity_at_points(
            points=vortex_centers,
            vortex_strengths=self.vortex_strengths,
        )
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        Vi_cross_li = np.cross(velocities, vortex_bound_leg, axis=1)

        forces_inviscid_geometry = (
            self.op_point.atmosphere.density()
            * Vi_cross_li
            * tall(self.vortex_strengths)
        )
        moments_inviscid_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_inviscid_geometry,
        )
        force_inviscid_geometry = np.sum(forces_inviscid_geometry, axis=0)
        moment_inviscid_geometry = np.sum(moments_inviscid_geometry, axis=0)

        if solve:
            residuals = self.sol(residuals)
            local_alphas = self.sol(local_alphas)
            local_Res = self.sol(local_Res)
            local_machs = self.sol(local_machs)
            CL_from_gamma = self.sol(CL_from_gamma)
            CL_2D = self.sol(CL_2D)
            CD_2D = self.sol(CD_2D)
            CM_2D = self.sol(CM_2D)
            analysis_confidences = self.sol(analysis_confidences)

        forces_profile_geometry = (
            0.5
            * self.op_point.atmosphere.density()
            * velocities
            * tall(velocity_magnitudes)
            * tall(CD_2D)
            * tall(areas)
        )
        moments_profile_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_profile_geometry,
        )
        force_profile_geometry = np.sum(forces_profile_geometry, axis=0)
        moment_profile_geometry = np.sum(moments_profile_geometry, axis=0)

        force_total_geometry = force_inviscid_geometry + force_profile_geometry
        moment_total_geometry = moment_inviscid_geometry + moment_profile_geometry

        force_inviscid_wind = np.array(
            self.op_point.convert_axes(
                force_inviscid_geometry[0],
                force_inviscid_geometry[1],
                force_inviscid_geometry[2],
                from_axes="geometry",
                to_axes="wind",
            )
        )
        force_profile_wind = np.array(
            self.op_point.convert_axes(
                force_profile_geometry[0],
                force_profile_geometry[1],
                force_profile_geometry[2],
                from_axes="geometry",
                to_axes="wind",
            )
        )

        force_total_body = np.array(
            self.op_point.convert_axes(
                force_total_geometry[0],
                force_total_geometry[1],
                force_total_geometry[2],
                from_axes="geometry",
                to_axes="body",
            )
        )
        force_total_wind = np.array(
            self.op_point.convert_axes(
                force_total_geometry[0],
                force_total_geometry[1],
                force_total_geometry[2],
                from_axes="geometry",
                to_axes="wind",
            )
        )
        moment_total_body = np.array(
            self.op_point.convert_axes(
                moment_total_geometry[0],
                moment_total_geometry[1],
                moment_total_geometry[2],
                from_axes="geometry",
                to_axes="body",
            )
        )
        moment_total_wind = np.array(
            self.op_point.convert_axes(
                moment_total_geometry[0],
                moment_total_geometry[1],
                moment_total_geometry[2],
                from_axes="geometry",
                to_axes="wind",
            )
        )

        L = -force_total_wind[2]
        D = -force_total_wind[0]
        Di = -force_inviscid_wind[0]
        Dp = -force_profile_wind[0]
        Y = force_total_wind[1]
        l_b = moment_total_body[0]
        m_b = moment_total_body[1]
        n_b = moment_total_body[2]

        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref

        CL = L / qS
        CD = D / qS
        CDi = Di / qS
        CDp = Dp / qS
        CY = Y / qS
        Cl = l_b / qS / b_ref
        Cm = m_b / qS / c_ref
        Cn = n_b / qS / b_ref

        section_lift = (
            0.5
            * self.op_point.atmosphere.density()
            * velocity_magnitudes**2
            * areas
            * CL_2D
        )
        section_drag_profile = (
            0.5
            * self.op_point.atmosphere.density()
            * velocity_magnitudes**2
            * areas
            * CD_2D
        )

        self.CL = CL
        self.CD = CD
        self.CDi = CDi
        self.CDp = CDp
        self.CY = CY
        self.Cl = Cl
        self.Cm = Cm
        self.Cn = Cn
        self.CL_over_CD = np.where(CD == 0, 0, np.array(CL / CD))

        self.residuals = residuals
        self.spanwise_y = spanwise_y
        self.local_alpha = local_alphas
        self.local_Re = local_Res
        self.local_mach = local_machs
        self.local_C_mu = local_C_mu
        self.local_x_jet = local_x_jet
        self.local_theta_jet = local_theta_jet
        self.section_CL_from_gamma = CL_from_gamma
        self.section_CL = CL_2D
        self.section_CD = CD_2D
        self.section_CM = CM_2D
        self.section_area = areas
        self.section_chord = chords
        self.section_lift = section_lift
        self.section_drag_profile = section_drag_profile
        self.section_analysis_confidence = analysis_confidences
        self.residual_CL = residuals
        self.residual_CM = np.zeros(np.length(residuals))
        self.max_residual_CL = np.max(np.abs(self.residual_CL))
        self.max_residual_CM = 0.0
        self.max_residual = self.max_residual_CL
        if solve:
            try:
                solver_stats = self.sol.stats()
                self.iteration_count = int(solver_stats.get("iter_count", 1))
            except Exception:
                self.iteration_count = 1
            self.converged = bool(float(self.max_residual_CL) <= self.residual_tolerance_CL)
            if not self.converged and failure_reason is None:
                failure_reason = (
                    f"residual_CL_above_tolerance: {float(self.max_residual_CL):.3g} > "
                    f"{self.residual_tolerance_CL:.3g}"
                )
        else:
            self.iteration_count = 0
            self.converged = False
        self.failure_reason = failure_reason
        self.convergence_history = [
            {
                "iteration": self.iteration_count,
                "max_abs_residual_CL": self.max_residual_CL,
                "rms_residual_CL": np.sqrt(np.mean(self.residual_CL**2)),
                "max_abs_residual_CM": self.max_residual_CM,
                "rms_residual_CM": 0.0,
                "max_residual": self.max_residual,
                "note": (
                    "Final implicit solve residual; per-optimizer-iteration residuals are not exposed by this path."
                ),
            }
        ]

        output = {
            "residuals": residuals,
            "residual_CL": self.residual_CL,
            "residual_CM": self.residual_CM,
            "gamma": self.vortex_strengths,
            "vortex_strengths": self.vortex_strengths,
            "F_g": force_total_geometry,
            "F_b": force_total_body,
            "F_w": force_total_wind,
            "M_g": moment_total_geometry,
            "M_b": moment_total_body,
            "M_w": moment_total_wind,
            "L": L,
            "D": D,
            "Y": Y,
            "l_b": l_b,
            "m_b": m_b,
            "n_b": n_b,
            "CL": CL,
            "CD": CD,
            "CDi": CDi,
            "CDp": CDp,
            "CY": CY,
            "CL_over_CD": self.CL_over_CD,
            "Cl": Cl,
            "Cm": Cm,
            "CM": Cm,
            "Cn": Cn,
            "spanwise_y": spanwise_y,
            "local_alpha": local_alphas,
            "local_Re": local_Res,
            "local_mach": local_machs,
            "local_C_mu": local_C_mu,
            "local_x_jet": local_x_jet,
            "local_theta_jet": local_theta_jet,
            "section_CL_from_gamma": CL_from_gamma,
            "section_CL": CL_2D,
            "section_CD": CD_2D,
            "section_CM": CM_2D,
            "section_cl": CL_2D,
            "section_cd": CD_2D,
            "section_cm": CM_2D,
            "section_area": areas,
            "section_chord": chords,
            "section_lift": section_lift,
            "section_drag_profile": section_drag_profile,
            "analysis_confidence": analysis_confidences,
            "convergence_history": self.convergence_history,
            "max_residual": self.max_residual,
            "iteration_count": self.iteration_count,
            "converged": self.converged,
            "failure_reason": self.failure_reason,
        }
        return add_afc_3d_postprocessing(output)


NeuralFoilAFCNonlinearLiftingLine = AFCNeuralFoilNonlinearLiftingLine
