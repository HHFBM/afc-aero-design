from aerosandbox.aerodynamics.aero_3D.afc_lifting_line import (
    AFCNeuralFoilLiftingLine,
)
from aerosandbox.aerodynamics.aero_3D.vortex_lattice_method import (
    VortexLatticeMethod,
    tall,
    wide,
)
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


class AFCNeuralFoilCamberedVLM(VortexLatticeMethod):
    """
    VLM with local virtual-camber normal-vector correction to match AFC-NeuralFoil section aerodynamics.

    This is a first-pass cambering-method implementation. The geometry and horseshoe vortex lattice are not rotated or
    moved. Instead, the local panel boundary-condition normals are adjusted by two spanwise virtual variables:

    - `delta_alpha_i`: constant normal rotation across the chord of a strip, used primarily to match section `CL`.
    - `delta_camber_i`: linear chordwise normal rotation, used primarily to match section `CM`.

    The update is explicit and under-relaxed. It is intended as a robust engineering scaffold before moving this model
    to a fully implicit optimization formulation.
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
        spanwise_resolution: int = 6,
        spanwise_spacing_function: Callable[
            [float, float, float], np.ndarray
        ] = np.cosspace,
        chordwise_resolution: int = 6,
        chordwise_spacing_function: Callable[
            [float, float, float], np.ndarray
        ] = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        include_360_deg_effects: bool = True,
        match_CM: bool = True,
        max_iterations: int = 25,
        relaxation_CL: float = 0.45,
        relaxation_CM: float = 0.35,
        tolerance_CL: float = 1e-3,
        tolerance_CM: float = 2e-3,
        max_abs_delta_alpha: float = 20.0,
        max_abs_delta_camber: float = 20.0,
        cm_camber_slope_per_rad: float = -3.0,
    ):
        super().__init__(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            run_symmetric_if_possible=run_symmetric_if_possible,
            verbose=verbose,
            spanwise_resolution=spanwise_resolution,
            spanwise_spacing_function=spanwise_spacing_function,
            chordwise_resolution=chordwise_resolution,
            chordwise_spacing_function=chordwise_spacing_function,
            vortex_core_radius=vortex_core_radius,
            align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
        )
        self.C_mu = C_mu
        self.x_jet = x_jet
        self.theta_jet = theta_jet
        self.model_size = model_size
        self.weights_directory = weights_directory
        self.training_statistics_path = training_statistics_path
        self.include_360_deg_effects = include_360_deg_effects
        self.match_CM = match_CM
        self.max_iterations = max_iterations
        self.relaxation_CL = relaxation_CL
        self.relaxation_CM = relaxation_CM
        self.tolerance_CL = tolerance_CL
        self.tolerance_CM = tolerance_CM
        self.max_abs_delta_alpha = max_abs_delta_alpha
        self.max_abs_delta_camber = max_abs_delta_camber
        self.cm_camber_slope_per_rad = cm_camber_slope_per_rad

    @staticmethod
    def _rotate_vectors_about_axes(vectors, axes, angles_deg):
        angles_rad = np.radians(angles_deg)
        c = tall(np.cos(angles_rad))
        s = tall(np.sin(angles_rad))
        return (
            vectors * c
            + np.cross(axes, vectors, axis=1) * s
            + axes * tall(np.sum(axes * vectors, axis=1)) * (1 - c)
        )

    def _mesh(self) -> Dict[str, object]:
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        strip_indices = []
        chord_indices = []
        airfoils: List[Airfoil] = []
        control_surfaces: List[List[ControlSurface]] = []

        strip_offset = 0
        for wing in self.airplane.wings:
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing_function=self.chordwise_spacing_function,
                add_camber=True,
            )
            n_faces = len(faces)
            n_strips = n_faces // self.chordwise_resolution

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])
            strip_indices.append(
                strip_offset
                + np.floor(np.arange(n_faces) / self.chordwise_resolution).astype(int)
            )
            chord_indices.append(np.arange(n_faces) % self.chordwise_resolution)

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

            repeats = max(1, n_strips // max(1, len(wing_airfoils)))
            for _ in range(repeats):
                airfoils.extend(wing_airfoils)
                control_surfaces.extend(wing_control_surfaces)

            strip_offset += n_strips

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        strip_indices = np.concatenate(strip_indices)
        chord_indices = np.concatenate(chord_indices)

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
        collocation_points = 0.5 * (
            0.25 * front_left_vertices + 0.75 * back_left_vertices
        ) + 0.5 * (0.25 * front_right_vertices + 0.75 * back_right_vertices)
        chord_vectors = (back_left_vertices + back_right_vertices) / 2 - (
            front_left_vertices + front_right_vertices
        ) / 2
        chords = np.linalg.norm(chord_vectors, axis=1)
        span_axes = vortex_bound_leg / tall(vortex_bound_leg_norm)
        local_forward_direction = np.cross(normal_directions, span_axes)

        n_strips_total = int(np.max(strip_indices)) + 1
        chordwise_fraction = (chord_indices + 0.5) / self.chordwise_resolution
        camber_shape = 2 * chordwise_fraction - 1

        return dict(
            front_left_vertices=front_left_vertices,
            back_left_vertices=back_left_vertices,
            back_right_vertices=back_right_vertices,
            front_right_vertices=front_right_vertices,
            normal_directions=normal_directions,
            areas=areas,
            left_vortex_vertices=left_vortex_vertices,
            right_vortex_vertices=right_vortex_vertices,
            vortex_centers=vortex_centers,
            vortex_bound_leg=vortex_bound_leg,
            collocation_points=collocation_points,
            chords=chords,
            span_axes=span_axes,
            local_forward_direction=local_forward_direction,
            strip_indices=strip_indices,
            chord_indices=chord_indices,
            camber_shape=camber_shape,
            n_strips=n_strips_total,
            airfoils=airfoils,
            control_surfaces=control_surfaces,
        )

    def _induced_velocity_at_points(self, mesh, points, vortex_strengths):
        u, v, w = calculate_induced_velocity_horseshoe(
            x_field=tall(points[:, 0]),
            y_field=tall(points[:, 1]),
            z_field=tall(points[:, 2]),
            x_left=wide(mesh["left_vortex_vertices"][:, 0]),
            y_left=wide(mesh["left_vortex_vertices"][:, 1]),
            z_left=wide(mesh["left_vortex_vertices"][:, 2]),
            x_right=wide(mesh["right_vortex_vertices"][:, 0]),
            y_right=wide(mesh["right_vortex_vertices"][:, 1]),
            z_right=wide(mesh["right_vortex_vertices"][:, 2]),
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1, 0, 0])
            ),
            gamma=wide(vortex_strengths),
            vortex_core_radius=self.vortex_core_radius,
        )
        return np.stack([np.sum(u, axis=1), np.sum(v, axis=1), np.sum(w, axis=1)], axis=1)

    def _velocity_at_points(self, mesh, points, vortex_strengths):
        return (
            self._induced_velocity_at_points(mesh, points, vortex_strengths)
            + np.tile(wide(self.steady_freestream_velocity), reps=(len(points), 1))
            + self.op_point.compute_rotation_velocity_geometry_axes(points)
        )

    def _solve_once(self, mesh, delta_alpha, delta_camber):
        strip_indices = mesh["strip_indices"].astype(int)
        panel_delta = (
            delta_alpha[strip_indices]
            + delta_camber[strip_indices] * mesh["camber_shape"]
        )
        corrected_normals = self._rotate_vectors_about_axes(
            mesh["normal_directions"],
            mesh["span_axes"],
            panel_delta,
        )
        corrected_normals = corrected_normals / tall(
            np.linalg.norm(corrected_normals, axis=1)
        )

        freestream_velocities = (
            np.tile(
                wide(self.steady_freestream_velocity),
                reps=(len(mesh["collocation_points"]), 1),
            )
            + self.op_point.compute_rotation_velocity_geometry_axes(
                mesh["collocation_points"]
            )
        )
        freestream_influences = np.sum(freestream_velocities * corrected_normals, axis=1)

        u, v, w = calculate_induced_velocity_horseshoe(
            x_field=tall(mesh["collocation_points"][:, 0]),
            y_field=tall(mesh["collocation_points"][:, 1]),
            z_field=tall(mesh["collocation_points"][:, 2]),
            x_left=wide(mesh["left_vortex_vertices"][:, 0]),
            y_left=wide(mesh["left_vortex_vertices"][:, 1]),
            z_left=wide(mesh["left_vortex_vertices"][:, 2]),
            x_right=wide(mesh["right_vortex_vertices"][:, 0]),
            y_right=wide(mesh["right_vortex_vertices"][:, 1]),
            z_right=wide(mesh["right_vortex_vertices"][:, 2]),
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1, 0, 0])
            ),
            gamma=1.0,
            vortex_core_radius=self.vortex_core_radius,
        )
        AIC = (
            u * tall(corrected_normals[:, 0])
            + v * tall(corrected_normals[:, 1])
            + w * tall(corrected_normals[:, 2])
        )
        vortex_strengths = np.linalg.solve(AIC, -freestream_influences)

        velocities = self._velocity_at_points(
            mesh, mesh["vortex_centers"], vortex_strengths
        )
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        forces_inviscid_geometry = (
            self.op_point.atmosphere.density()
            * np.cross(velocities, mesh["vortex_bound_leg"], axis=1)
            * tall(vortex_strengths)
        )

        return dict(
            vortex_strengths=vortex_strengths,
            corrected_normals=corrected_normals,
            velocities=velocities,
            velocity_magnitudes=velocity_magnitudes,
            forces_inviscid_geometry=forces_inviscid_geometry,
        )

    def _strip_outputs(self, mesh, solve_data, delta_camber):
        n_strips = mesh["n_strips"]
        strip_indices = mesh["strip_indices"].astype(int)
        spanwise_y = np.array(
            [
                np.mean(mesh["vortex_centers"][strip_indices == i, 1])
                for i in range(n_strips)
            ]
        )
        section_area = np.array(
            [np.sum(mesh["areas"][strip_indices == i]) for i in range(n_strips)]
        )
        section_chord = np.array(
            [np.mean(mesh["chords"][strip_indices == i]) for i in range(n_strips)]
        )
        section_velocity = np.array(
            [
                np.mean(solve_data["velocities"][strip_indices == i, :], axis=0)
                for i in range(n_strips)
            ]
        )
        section_velocity_magnitude = np.linalg.norm(section_velocity, axis=1)
        section_velocity_direction = section_velocity / tall(section_velocity_magnitude)
        section_normal = np.array(
            [
                np.mean(mesh["normal_directions"][strip_indices == i, :], axis=0)
                for i in range(n_strips)
            ]
        )
        section_normal = section_normal / tall(np.linalg.norm(section_normal, axis=1))
        section_forward = np.array(
            [
                np.mean(mesh["local_forward_direction"][strip_indices == i, :], axis=0)
                for i in range(n_strips)
            ]
        )
        section_forward = section_forward / tall(np.linalg.norm(section_forward, axis=1))

        local_alpha = 90 - np.arccosd(
            np.sum(section_velocity_direction * section_normal, axis=1)
        )
        cos_sweep = np.sum(section_velocity_direction * -section_forward, axis=1)
        local_Re = (
            section_velocity_magnitude
            * section_chord
            / self.op_point.atmosphere.kinematic_viscosity()
        ) * cos_sweep
        local_mach = (
            section_velocity_magnitude
            / self.op_point.atmosphere.speed_of_sound()
            * cos_sweep
        )

        local_C_mu = AFCNeuralFoilLiftingLine._spanwise_distribution(
            self.C_mu, spanwise_y, "C_mu"
        )
        local_x_jet = AFCNeuralFoilLiftingLine._spanwise_distribution(
            self.x_jet, spanwise_y, "x_jet"
        )
        local_theta_jet = AFCNeuralFoilLiftingLine._spanwise_distribution(
            self.theta_jet, spanwise_y, "theta_jet"
        )

        section_aero = [
            mesh["airfoils"][i].get_aero_from_afc_neuralfoil(
                alpha=local_alpha[i],
                Re=local_Re[i],
                mach=local_mach[i],
                C_mu=local_C_mu[i],
                x_jet=local_x_jet[i],
                theta_jet=local_theta_jet[i],
                control_surfaces=mesh["control_surfaces"][i],
                model_size=self.model_size,
                weights_directory=self.weights_directory,
                training_statistics_path=self.training_statistics_path,
                include_360_deg_effects=self.include_360_deg_effects,
            )
            for i in range(n_strips)
        ]
        section_CL_2D = np.reshape(np.array([a["CL"] for a in section_aero]), -1)
        section_CD_2D = np.reshape(np.array([a["CD"] for a in section_aero]), -1)
        section_CM_2D = np.reshape(np.array([a["CM"] for a in section_aero]), -1)
        analysis_confidence = np.reshape(
            np.array([a["analysis_confidence"] for a in section_aero]), -1
        )

        forces_wind = np.array(
            self.op_point.convert_axes(
                solve_data["forces_inviscid_geometry"][:, 0],
                solve_data["forces_inviscid_geometry"][:, 1],
                solve_data["forces_inviscid_geometry"][:, 2],
                from_axes="geometry",
                to_axes="wind",
            )
        ).T
        section_lift = np.array(
            [-np.sum(forces_wind[strip_indices == i, 2]) for i in range(n_strips)]
        )
        q_section = 0.5 * self.op_point.atmosphere.density() * section_velocity_magnitude**2
        section_CL_vlm = section_lift / q_section / section_area

        section_quarter_chord = np.array(
            [
                np.mean(
                    0.75 * mesh["front_left_vertices"][strip_indices == i, :]
                    + 0.25 * mesh["back_left_vertices"][strip_indices == i, :],
                    axis=0,
                )
                for i in range(n_strips)
            ]
        )
        section_CM_vlm_raw = []
        for i in range(n_strips):
            mask = strip_indices == i
            moments = np.cross(
                mesh["vortex_centers"][mask, :] - wide(section_quarter_chord[i, :]),
                solve_data["forces_inviscid_geometry"][mask, :],
            )
            moment_body = self.op_point.convert_axes(
                np.sum(moments[:, 0]),
                np.sum(moments[:, 1]),
                np.sum(moments[:, 2]),
                from_axes="geometry",
                to_axes="body",
            )
            section_CM_vlm_raw.append(
                moment_body[1] / q_section[i] / section_chord[i] / section_area[i]
            )
        section_CM_vlm_raw = np.array(section_CM_vlm_raw)
        section_CM_vlm = (
            section_CM_vlm_raw
            + self.cm_camber_slope_per_rad * np.radians(delta_camber)
        )

        return dict(
            spanwise_y=spanwise_y,
            section_area=section_area,
            section_chord=section_chord,
            section_lift=section_lift,
            section_drag_profile=q_section * section_area * section_CD_2D,
            local_alpha=local_alpha,
            local_Re=local_Re,
            local_mach=local_mach,
            local_C_mu=local_C_mu,
            local_x_jet=local_x_jet,
            local_theta_jet=local_theta_jet,
            section_CL=section_CL_2D,
            section_CD=section_CD_2D,
            section_CM=section_CM_2D,
            section_CL_vlm=section_CL_vlm,
            section_CM_vlm_raw=section_CM_vlm_raw,
            section_CM_vlm=section_CM_vlm,
            residual_CL=section_CL_vlm - section_CL_2D,
            residual_CM=section_CM_vlm - section_CM_2D,
            analysis_confidence=analysis_confidence,
        )

    def _aggregate_outputs(self, mesh, solve_data, strip_data):
        strip_indices = mesh["strip_indices"].astype(int)
        panel_CD = strip_data["section_CD"][strip_indices]
        forces_profile_geometry = (
            0.5
            * self.op_point.atmosphere.density()
            * solve_data["velocities"]
            * tall(solve_data["velocity_magnitudes"])
            * tall(panel_CD)
            * tall(mesh["areas"])
        )
        force_inviscid_geometry = np.sum(solve_data["forces_inviscid_geometry"], axis=0)
        force_profile_geometry = np.sum(forces_profile_geometry, axis=0)
        force_total_geometry = force_inviscid_geometry + force_profile_geometry

        moments_inviscid_geometry = np.cross(
            mesh["vortex_centers"] - wide(np.array(self.xyz_ref)),
            solve_data["forces_inviscid_geometry"],
        )
        moments_profile_geometry = np.cross(
            mesh["vortex_centers"] - wide(np.array(self.xyz_ref)),
            forces_profile_geometry,
        )
        moment_total_geometry = (
            np.sum(moments_inviscid_geometry, axis=0)
            + np.sum(moments_profile_geometry, axis=0)
        )
        delta_CM = strip_data["section_CM_vlm"] - strip_data["section_CM_vlm_raw"]
        section_span_width = strip_data["section_area"] / strip_data["section_chord"]
        section_span_axis = np.array(
            [
                np.mean(mesh["span_axes"][strip_indices == i, :], axis=0)
                for i in range(mesh["n_strips"])
            ]
        )
        section_span_axis = section_span_axis / tall(
            np.linalg.norm(section_span_axis, axis=1)
        )
        section_span_vectors = section_span_axis * tall(section_span_width)
        section_span_vectors = np.stack(
            [
                np.zeros(mesh["n_strips"]),
                section_span_vectors[:, 1],
                section_span_vectors[:, 2],
            ],
            axis=1,
        )
        q_section = (
            0.5
            * self.op_point.atmosphere.density()
            * (
                np.array(
                    [
                        np.linalg.norm(
                            np.mean(
                                solve_data["velocities"][strip_indices == i, :],
                                axis=0,
                            )
                        )
                        for i in range(mesh["n_strips"])
                    ]
                )
                ** 2
            )
        )
        moment_delta_camber_geometry = np.sum(
            tall(q_section)
            * tall(delta_CM)
            * tall(strip_data["section_chord"] ** 2)
            * section_span_vectors,
            axis=0,
        )
        moment_total_geometry = moment_total_geometry + moment_delta_camber_geometry

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

        qS = self.op_point.dynamic_pressure() * self.airplane.s_ref
        L = -force_total_wind[2]
        D = -force_total_wind[0]
        Di = -force_inviscid_wind[0]
        Dp = -force_profile_wind[0]
        Y = force_total_wind[1]
        l_b = moment_total_body[0]
        m_b = moment_total_body[1]
        n_b = moment_total_body[2]

        return dict(
            F_g=force_total_geometry,
            F_b=force_total_body,
            F_w=force_total_wind,
            M_g=moment_total_geometry,
            M_b=moment_total_body,
            M_w=moment_total_wind,
            L=L,
            D=D,
            Y=Y,
            l_b=l_b,
            m_b=m_b,
            n_b=n_b,
            CL=L / qS,
            CD=D / qS,
            CDi=Di / qS,
            CDp=Dp / qS,
            CY=Y / qS,
            Cl=l_b / qS / self.airplane.b_ref,
            Cm=m_b / qS / self.airplane.c_ref,
            CM=m_b / qS / self.airplane.c_ref,
            Cn=n_b / qS / self.airplane.b_ref,
        )

    def run(self) -> Dict[str, object]:
        mesh = self._mesh()

        for key, value in mesh.items():
            if key not in {"airfoils", "control_surfaces", "n_strips"}:
                setattr(self, key, value)
        self.airfoils = mesh["airfoils"]
        self.control_surfaces = mesh["control_surfaces"]
        self.n_panels = len(mesh["areas"])
        self.n_strips = mesh["n_strips"]

        self.steady_freestream_velocity = (
            self.op_point.compute_freestream_velocity_geometry_axes()
        )
        self.steady_freestream_direction = self.steady_freestream_velocity / np.linalg.norm(
            self.steady_freestream_velocity
        )

        delta_alpha = np.zeros(self.n_strips)
        delta_camber = np.zeros(self.n_strips)
        convergence_history = []

        solve_data = None
        strip_data = None
        for iteration in range(self.max_iterations):
            solve_data = self._solve_once(mesh, delta_alpha, delta_camber)
            strip_data = self._strip_outputs(mesh, solve_data, delta_camber)

            max_res_CL = float(np.max(np.abs(strip_data["residual_CL"])))
            max_res_CM = float(np.max(np.abs(strip_data["residual_CM"])))
            convergence_history.append(
                dict(
                    iteration=iteration,
                    max_abs_residual_CL=max_res_CL,
                    rms_residual_CL=float(
                        np.sqrt(np.mean(strip_data["residual_CL"] ** 2))
                    ),
                    max_abs_residual_CM=max_res_CM,
                    rms_residual_CM=float(
                        np.sqrt(np.mean(strip_data["residual_CM"] ** 2))
                    ),
                    max_residual=max(max_res_CL, max_res_CM if self.match_CM else 0.0),
                    max_abs_delta_alpha=float(np.max(np.abs(delta_alpha))),
                    max_abs_delta_camber=float(np.max(np.abs(delta_camber))),
                )
            )

            CL_converged = max_res_CL < self.tolerance_CL
            CM_converged = (not self.match_CM) or max_res_CM < self.tolerance_CM
            if CL_converged and CM_converged:
                break

            delta_alpha = delta_alpha - self.relaxation_CL * np.degrees(
                strip_data["residual_CL"] / (2 * np.pi)
            )
            delta_alpha = np.clip(
                delta_alpha, -self.max_abs_delta_alpha, self.max_abs_delta_alpha
            )

            if self.match_CM:
                delta_camber = delta_camber - self.relaxation_CM * np.degrees(
                    strip_data["residual_CM"] / self.cm_camber_slope_per_rad
                )
                delta_camber = np.clip(
                    delta_camber,
                    -self.max_abs_delta_camber,
                    self.max_abs_delta_camber,
                )

        self.vortex_strengths = solve_data["vortex_strengths"]
        self.delta_alpha = delta_alpha
        self.delta_camber = delta_camber
        self.convergence_history = convergence_history
        self.iteration_count = len(convergence_history)
        self.max_residual = convergence_history[-1]["max_residual"]
        self.converged_CL = convergence_history[-1]["max_abs_residual_CL"] < self.tolerance_CL
        self.converged_CM = (not self.match_CM) or (
            convergence_history[-1]["max_abs_residual_CM"] < self.tolerance_CM
        )
        self.converged = bool(self.converged_CL and self.converged_CM)
        self.failure_reason = None if self.converged else (
            "cambered_vlm_residual_not_converged: "
            f"max_CL={convergence_history[-1]['max_abs_residual_CL']:.3g}, "
            f"max_CM={convergence_history[-1]['max_abs_residual_CM']:.3g}, "
            f"iterations={self.iteration_count}/{self.max_iterations}"
        )

        aggregate = self._aggregate_outputs(mesh, solve_data, strip_data)
        output = {
            **aggregate,
            **strip_data,
            "section_cl": strip_data["section_CL"],
            "section_cd": strip_data["section_CD"],
            "section_cm": strip_data["section_CM"],
            "vortex_strengths": self.vortex_strengths,
            "delta_alpha": delta_alpha,
            "delta_camber": delta_camber,
            "residual_CL": strip_data["residual_CL"],
            "residual_CM": strip_data["residual_CM"],
            "convergence_history": convergence_history,
            "max_residual": self.max_residual,
            "iteration_count": self.iteration_count,
            "converged_CL": self.converged_CL,
            "converged_CM": self.converged_CM,
            "converged": self.converged,
            "failure_reason": self.failure_reason,
        }

        self.CL = output["CL"]
        self.CD = output["CD"]
        self.CDi = output["CDi"]
        self.CDp = output["CDp"]
        self.CY = output["CY"]
        self.Cl = output["Cl"]
        self.Cm = output["Cm"]
        self.Cn = output["Cn"]
        self.CM = output["CM"]
        return add_afc_3d_postprocessing(output)


NeuralFoilCamberedVLM = AFCNeuralFoilCamberedVLM
CamberedVLM = AFCNeuralFoilCamberedVLM
