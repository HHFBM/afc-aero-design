from aerosandbox.aerodynamics.aero_3D.lifting_line import LiftingLine, tall, wide
from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import (
    calculate_induced_velocity_horseshoe,
)
from aerosandbox.aerodynamics.aero_3D.afc_postprocessing import (
    add_afc_3d_postprocessing,
)
from aerosandbox.geometry import *
from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
from typing import Callable, Dict, List, Union


class AFCNeuralFoilLiftingLine(LiftingLine):
    """
    A first-pass finite-wing AFC analysis built on AeroSandbox's existing LiftingLine machinery.

    This analysis reuses AeroSandbox wing geometry, thin-surface meshing, horseshoe-vortex induced velocity, and
    near-field force integration. The only AFC-specific step is the local 2D section model: each spanwise section calls
    `Airfoil.get_aero_from_afc_neuralfoil()`.

    The AFC correction is currently a placeholder empirical surrogate inherited from the 2D method, not a trained
    NeuralFoil model. The class is structured so the section model can be replaced later without changing the 3D data
    flow.
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
    ):
        super().__init__(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            model_size=model_size,
            run_symmetric_if_possible=run_symmetric_if_possible,
            verbose=verbose,
            spanwise_resolution=spanwise_resolution,
            spanwise_spacing_function=spanwise_spacing_function,
            vortex_core_radius=vortex_core_radius,
            align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
        )

        self.C_mu = C_mu
        self.x_jet = x_jet
        self.theta_jet = theta_jet
        self.weights_directory = weights_directory
        self.training_statistics_path = training_statistics_path
        self.include_360_deg_effects = include_360_deg_effects

    @staticmethod
    def _spanwise_distribution(
        value: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]],
        spanwise_y: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """
        Expands a scalar, callable, per-panel array, or semi-span distribution onto each spanwise panel.
        """
        if callable(value):
            return value(spanwise_y)

        value = np.array(value)
        n_panels = np.length(spanwise_y)
        n_values = np.length(value)

        if n_values == 1:
            return np.ones(n_panels) * value
        elif n_values == n_panels:
            return np.reshape(value, (-1,))
        else:
            eta = np.abs(spanwise_y)
            eta = (eta - np.min(eta)) / np.softmax(np.max(eta) - np.min(eta), 1e-12)
            eta = np.reshape(eta, (-1,))
            xp = np.linspace(0, 1, n_values)
            fp = np.reshape(value, (-1,))
            return np.array(
                [
                    np.interp(
                        eta[i],
                        xp,
                        fp,
                    )
                    for i in range(n_panels)
                ]
            )

    def run(self) -> Dict[str, Union[float, np.ndarray, List[Union[float, np.ndarray]]]]:
        """
        Computes AFC finite-wing aerodynamic coefficients and spanwise section diagnostics.

        Returns:
            A dictionary containing aggregate coefficients (`CL`, `CD`, `CDi`, `CDp`, `CM`) and spanwise arrays:
            `spanwise_y`, `local_alpha`, `local_Re`, `local_C_mu`, `section_CL`, `section_CD`, `section_CM`,
            and `analysis_confidence`.
        """
        wing_aero = self.wing_aerodynamics()

        F_g_total = wing_aero.F_g
        M_g_total = wing_aero.M_g

        output = {
            "F_g": F_g_total,
            "M_g": M_g_total,
        }

        output["F_b"] = self.op_point.convert_axes(
            F_g_total[0],
            F_g_total[1],
            F_g_total[2],
            from_axes="geometry",
            to_axes="body",
        )
        output["F_w"] = self.op_point.convert_axes(
            F_g_total[0],
            F_g_total[1],
            F_g_total[2],
            from_axes="geometry",
            to_axes="wind",
        )
        output["M_b"] = self.op_point.convert_axes(
            M_g_total[0],
            M_g_total[1],
            M_g_total[2],
            from_axes="geometry",
            to_axes="body",
        )
        output["M_w"] = self.op_point.convert_axes(
            M_g_total[0],
            M_g_total[1],
            M_g_total[2],
            from_axes="geometry",
            to_axes="wind",
        )

        output["L"] = -output["F_w"][2]
        output["Y"] = output["F_w"][1]
        output["D"] = -output["F_w"][0]
        output["l_b"] = output["M_b"][0]
        output["m_b"] = output["M_b"][1]
        output["n_b"] = output["M_b"][2]

        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref
        c = self.airplane.c_ref
        b = self.airplane.b_ref

        output["CL"] = output["L"] / qS
        output["CY"] = output["Y"] / qS
        output["CD"] = output["D"] / qS
        output["Cl"] = output["l_b"] / qS / b
        output["Cm"] = output["m_b"] / qS / c
        output["Cn"] = output["n_b"] / qS / b
        output["CM"] = output["Cm"]

        output["CDi"] = self.CDi
        output["CDp"] = self.CDp

        output["spanwise_y"] = self.spanwise_y
        output["local_alpha"] = self.local_alpha
        output["local_Re"] = self.local_Re
        output["local_C_mu"] = self.local_C_mu
        output["section_CL"] = self.section_CL
        output["section_CD"] = self.section_CD
        output["section_CM"] = self.section_CM
        output["section_cl"] = self.section_CL
        output["section_cd"] = self.section_CD
        output["section_cm"] = self.section_CM
        output["section_area"] = self.section_area
        output["section_chord"] = self.section_chord
        output["section_lift"] = self.section_lift
        output["section_drag_profile"] = self.section_drag_profile
        output["analysis_confidence"] = self.section_analysis_confidence
        output["residual_CL"] = self.residual_CL
        output["residual_CM"] = self.residual_CM
        output["convergence_history"] = self.convergence_history
        output["max_residual"] = self.max_residual
        output["iteration_count"] = self.iteration_count
        output["converged"] = self.converged
        output["failure_reason"] = self.failure_reason
        output["wing_aero"] = wing_aero

        return add_afc_3d_postprocessing(output)

    def wing_aerodynamics(self) -> LiftingLine.AeroComponentResults:
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
        local_C_mu = self._spanwise_distribution(self.C_mu, spanwise_y, "C_mu")
        local_x_jet = self._spanwise_distribution(self.x_jet, spanwise_y, "x_jet")
        local_theta_jet = self._spanwise_distribution(
            self.theta_jet, spanwise_y, "theta_jet"
        )

        if self.verbose:
            print("Calculating the freestream influence...")
        steady_freestream_velocity = self.op_point.compute_freestream_velocity_geometry_axes()
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(
            steady_freestream_velocity
        )
        steady_freestream_velocities = np.tile(
            wide(steady_freestream_velocity), reps=(self.n_panels, 1)
        )
        steady_freestream_directions = np.tile(
            wide(steady_freestream_direction), reps=(self.n_panels, 1)
        )
        rotation_freestream_velocities = (
            self.op_point.compute_rotation_velocity_geometry_axes(points=vortex_centers)
        )
        freestream_velocities = steady_freestream_velocities + rotation_freestream_velocities

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        alpha_geometrics = 90 - np.arccosd(
            np.sum(steady_freestream_directions * normal_directions, axis=1)
        )
        cos_sweeps = np.sum(
            steady_freestream_directions * -local_forward_direction, axis=1
        )
        machs = self.op_point.mach() * cos_sweeps
        Res = (
            self.op_point.velocity
            * chords
            / self.op_point.atmosphere.kinematic_viscosity()
        ) * cos_sweeps

        CLs_at_alpha_geometric = [
            af.get_aero_from_afc_neuralfoil(
                alpha=alpha_geometrics[i],
                Re=Res[i],
                mach=machs[i],
                C_mu=local_C_mu[i],
                x_jet=local_x_jet[i],
                theta_jet=local_theta_jet[i],
                control_surfaces=control_surfaces[i],
                model_size=self.model_size,
                weights_directory=self.weights_directory,
                training_statistics_path=self.training_statistics_path,
                include_360_deg_effects=self.include_360_deg_effects,
            )["CL"]
            for i, af in enumerate(airfoils)
        ]
        CLs_at_alpha_geometric = np.reshape(np.array(CLs_at_alpha_geometric), -1)
        CLas = 2 * np.pi * np.ones(self.n_panels)

        if self.verbose:
            print("Calculating the collocation influence matrix...")
        u_centers_unit, v_centers_unit, w_centers_unit = (
            calculate_induced_velocity_horseshoe(
                x_field=tall(vortex_centers[:, 0]),
                y_field=tall(vortex_centers[:, 1]),
                z_field=tall(vortex_centers[:, 2]),
                x_left=wide(left_vortex_vertices[:, 0]),
                y_left=wide(left_vortex_vertices[:, 1]),
                z_left=wide(left_vortex_vertices[:, 2]),
                x_right=wide(right_vortex_vertices[:, 0]),
                y_right=wide(right_vortex_vertices[:, 1]),
                z_right=wide(right_vortex_vertices[:, 2]),
                trailing_vortex_direction=(
                    steady_freestream_direction
                    if self.align_trailing_vortices_with_wind
                    else np.array([1, 0, 0])
                ),
                gamma=1.0,
                vortex_core_radius=self.vortex_core_radius,
            )
        )
        AIC = (
            u_centers_unit * tall(normal_directions[:, 0])
            + v_centers_unit * tall(normal_directions[:, 1])
            + w_centers_unit * tall(normal_directions[:, 2])
        )
        alpha_influence_matrix = AIC / self.op_point.velocity

        if self.verbose:
            print("Calculating vortex center strengths.")
        V_freestream_cross_li = np.cross(
            steady_freestream_velocities, self.vortex_bound_leg, axis=1
        )
        V_freestream_cross_li_magnitudes = np.linalg.norm(V_freestream_cross_li, axis=1)
        velocity_magnitude_perpendiculars = self.op_point.velocity * cos_sweeps

        A = alpha_influence_matrix * np.tile(wide(CLas), (self.n_panels, 1)) - np.diag(
            2
            * V_freestream_cross_li_magnitudes
            / velocity_magnitude_perpendiculars**2
            / areas
        )
        b = -1 * CLs_at_alpha_geometric
        vortex_strengths = np.linalg.solve(A, b)
        self.vortex_strengths = vortex_strengths

        alpha_induced = np.degrees(alpha_influence_matrix @ vortex_strengths)
        alphas = alpha_geometrics + alpha_induced

        aeros = [
            af.get_aero_from_afc_neuralfoil(
                alpha=alphas[i],
                Re=Res[i],
                mach=machs[i],
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
        CLs = np.reshape(np.array([aero["CL"] for aero in aeros]), -1)
        CDs = np.reshape(np.array([aero["CD"] for aero in aeros]), -1)
        CMs = np.reshape(np.array([aero["CM"] for aero in aeros]), -1)
        analysis_confidences = np.reshape(
            np.array([aero["analysis_confidence"] for aero in aeros]), -1
        )

        velocities = self.get_velocity_at_points(
            points=vortex_centers, vortex_strengths=vortex_strengths
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

        forces_profile_geometry = (
            0.5
            * self.op_point.atmosphere.density()
            * velocities
            * tall(velocity_magnitudes)
            * tall(CDs)
            * tall(areas)
        )
        moments_profile_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_profile_geometry,
        )
        force_profile_geometry = np.sum(forces_profile_geometry, axis=0)
        moment_profile_geometry = np.sum(moments_profile_geometry, axis=0)

        bound_leg_YZ = np.stack(
            [
                np.zeros(self.n_panels),
                vortex_bound_leg[:, 1],
                vortex_bound_leg[:, 2],
            ],
            axis=1,
        )
        moments_pitching_geometry = (
            (0.5 * self.op_point.atmosphere.density() * tall(velocity_magnitudes**2))
            * tall(CMs)
            * tall(chords**2)
            * bound_leg_YZ
        )
        moment_pitching_geometry = np.sum(moments_pitching_geometry, axis=0)

        force_total_geometry = np.add(force_inviscid_geometry, force_profile_geometry)
        moment_total_geometry = (
            np.add(moment_inviscid_geometry, moment_profile_geometry)
            + moment_pitching_geometry
        )

        force_inviscid_wind = self.op_point.convert_axes(
            force_inviscid_geometry[0],
            force_inviscid_geometry[1],
            force_inviscid_geometry[2],
            from_axes="geometry",
            to_axes="wind",
        )
        force_profile_wind = self.op_point.convert_axes(
            force_profile_geometry[0],
            force_profile_geometry[1],
            force_profile_geometry[2],
            from_axes="geometry",
            to_axes="wind",
        )
        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref
        self.CDi = -force_inviscid_wind[0] / qS
        self.CDp = -force_profile_wind[0] / qS

        self.spanwise_y = spanwise_y
        self.local_alpha = alphas
        self.local_Re = Res
        self.local_C_mu = local_C_mu
        self.section_CL = CLs
        self.section_CD = CDs
        self.section_CM = CMs
        self.section_area = areas
        self.section_chord = chords
        self.section_lift = (
            0.5
            * self.op_point.atmosphere.density()
            * velocity_magnitudes**2
            * areas
            * CLs
        )
        self.section_drag_profile = (
            0.5
            * self.op_point.atmosphere.density()
            * velocity_magnitudes**2
            * areas
            * CDs
        )
        self.section_analysis_confidence = analysis_confidences
        self.residual_CL = np.zeros(self.n_panels)
        self.residual_CM = np.zeros(self.n_panels)
        self.max_residual = 0.0
        self.iteration_count = 1
        self.converged = True
        self.failure_reason = None
        self.convergence_history = [
            {
                "iteration": 0,
                "max_abs_residual_CL": 0.0,
                "rms_residual_CL": 0.0,
                "max_abs_residual_CM": 0.0,
                "rms_residual_CM": 0.0,
                "max_residual": 0.0,
                "note": "Explicit lifting-line solve; no nonlinear residual iteration is performed.",
            }
        ]

        return self.AeroComponentResults(
            s_ref=self.airplane.s_ref,
            c_ref=self.airplane.c_ref,
            b_ref=self.airplane.b_ref,
            op_point=self.op_point,
            F_g=[force_total_geometry[i] for i in range(3)],
            M_g=[moment_total_geometry[i] for i in range(3)],
        )


NeuralFoilAFCLiftingLine = AFCNeuralFoilLiftingLine
