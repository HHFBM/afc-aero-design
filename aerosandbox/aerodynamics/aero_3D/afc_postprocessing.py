from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import aerosandbox.numpy as np


@dataclass
class AFC3DPostProcessResult:
    """
    Standard post-processing schema for 3D AFC finite-wing results.

    `afc_power_proxy` uses the engineering proxy `integral(C_mu * chord dy)`, evaluated as
    `sum(local_C_mu * section_area)` when section areas are available.
    """

    CL: Any
    CD: Any
    CDi: Any
    CDp: Any
    CM: Any
    CY: Any = None
    Cl: Any = None
    Cn: Any = None
    root_bending_moment: Any = None
    spanwise_y: Any = None
    spanwise_load: Any = None
    local_stall_indicator: Any = None
    analysis_confidence: Any = None
    afc_power_proxy: Any = None
    section_CL: Any = None
    section_CD: Any = None
    section_CM: Any = None
    local_alpha: Any = None
    local_Re: Any = None
    local_C_mu: Any = None
    residual_CL: Any = None
    residual_CM: Any = None
    convergence_history: Any = None
    max_residual: Any = None
    iteration_count: Any = None
    converged: Any = None
    failure_reason: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "CL": self.CL,
            "CD": self.CD,
            "CDi": self.CDi,
            "CDp": self.CDp,
            "CM": self.CM,
            "CY": self.CY,
            "Cl": self.Cl,
            "Cn": self.Cn,
            "root_bending_moment": self.root_bending_moment,
            "spanwise_y": self.spanwise_y,
            "spanwise_load": self.spanwise_load,
            "local_stall_indicator": self.local_stall_indicator,
            "analysis_confidence": self.analysis_confidence,
            "afc_power_proxy": self.afc_power_proxy,
            "section_CL": self.section_CL,
            "section_CD": self.section_CD,
            "section_CM": self.section_CM,
            "local_alpha": self.local_alpha,
            "local_Re": self.local_Re,
            "local_C_mu": self.local_C_mu,
            "residual_CL": self.residual_CL,
            "residual_CM": self.residual_CM,
            "convergence_history": self.convergence_history,
            "max_residual": self.max_residual,
            "iteration_count": self.iteration_count,
            "converged": self.converged,
            "failure_reason": self.failure_reason,
        }


def _get(result: Dict[str, Any], key: str, default=None):
    return result[key] if key in result else default


def estimate_spanwise_widths(spanwise_y) -> Any:
    """
    Estimates a local spanwise width for each section from section-center y locations.

    This is primarily used for load plotting. Integrated quantities prefer `section_area` when available.
    """
    y = np.reshape(np.array(spanwise_y), (-1,))
    n = np.length(y)
    if n == 1:
        return np.ones(1)

    try:
        import numpy as onp

        y_np = onp.asarray(y, dtype=float)
        order = onp.argsort(y_np)
        y_sorted = y_np[order]
        edges = onp.empty(n + 1)
        edges[1:-1] = 0.5 * (y_sorted[:-1] + y_sorted[1:])
        edges[0] = y_sorted[0] - 0.5 * (y_sorted[1] - y_sorted[0])
        edges[-1] = y_sorted[-1] + 0.5 * (y_sorted[-1] - y_sorted[-2])
        widths_sorted = onp.diff(edges)
        widths = onp.empty(n)
        widths[order] = onp.maximum(onp.abs(widths_sorted), 1e-12)
        return np.array(widths)
    except Exception:
        # Symbolic fallback: use a uniform scale. This preserves shapes and keeps downstream expressions valid.
        return np.ones(n)


def local_stall_indicator(
    local_alpha=None,
    section_CL=None,
    analysis_confidence=None,
) -> Any:
    """
    Returns a smooth 0..1 local stall-risk proxy.

    The proxy is intentionally conservative and model-agnostic:
    - high absolute local angle of attack increases the indicator;
    - high absolute section CL increases the indicator;
    - low analysis confidence increases the indicator.
    """
    indicators = []
    if local_alpha is not None:
        indicators.append(0.5 + 0.5 * np.tanh((np.abs(local_alpha) - 14.0) / 3.0))
    if section_CL is not None:
        indicators.append(0.5 + 0.5 * np.tanh((np.abs(section_CL) - 1.25) / 0.25))
    if analysis_confidence is not None:
        indicators.append(1 - analysis_confidence)

    if len(indicators) == 0:
        return None

    indicator = indicators[0]
    for next_indicator in indicators[1:]:
        indicator = np.maximum(indicator, next_indicator)
    return np.clip(indicator, 0, 1)


def postprocess_afc_3d_result(result: Dict[str, Any]) -> AFC3DPostProcessResult:
    """
    Builds the standard AFC 3D post-processing schema from an analysis result dictionary.
    """
    spanwise_y = _get(result, "spanwise_y")
    section_lift = _get(result, "section_lift")
    section_area = _get(result, "section_area")
    section_chord = _get(result, "section_chord")
    local_C_mu = _get(result, "local_C_mu")
    section_CL = _get(result, "section_CL")
    section_CD = _get(result, "section_CD")
    section_CM = _get(result, "section_CM")
    local_alpha = _get(result, "local_alpha")
    local_Re = _get(result, "local_Re")
    analysis_confidence = _get(result, "analysis_confidence")

    if spanwise_y is None:
        spanwise_y = np.array([])

    if section_lift is None and section_area is not None and section_CL is not None:
        # Fallback for methods that do not expose a sectional force directly.
        section_lift = section_CL * section_area

    spanwise_width = estimate_spanwise_widths(spanwise_y)
    if section_lift is not None:
        spanwise_load = section_lift / spanwise_width
        root_bending_moment = 0.5 * np.sum(np.abs(spanwise_y) * section_lift)
    else:
        spanwise_load = None
        root_bending_moment = None

    if local_C_mu is not None:
        if section_area is not None:
            afc_power_proxy = np.sum(local_C_mu * section_area)
        elif section_chord is not None:
            afc_power_proxy = np.sum(local_C_mu * section_chord * spanwise_width)
        else:
            afc_power_proxy = np.sum(local_C_mu * spanwise_width)
    else:
        afc_power_proxy = 0.0

    stall_indicator = local_stall_indicator(
        local_alpha=local_alpha,
        section_CL=section_CL,
        analysis_confidence=analysis_confidence,
    )

    return AFC3DPostProcessResult(
        CL=_get(result, "CL"),
        CD=_get(result, "CD"),
        CDi=_get(result, "CDi"),
        CDp=_get(result, "CDp"),
        CM=_get(result, "CM", _get(result, "Cm")),
        CY=_get(result, "CY"),
        Cl=_get(result, "Cl"),
        Cn=_get(result, "Cn"),
        root_bending_moment=root_bending_moment,
        spanwise_y=spanwise_y,
        spanwise_load=spanwise_load,
        local_stall_indicator=stall_indicator,
        analysis_confidence=analysis_confidence,
        afc_power_proxy=afc_power_proxy,
        section_CL=section_CL,
        section_CD=section_CD,
        section_CM=section_CM,
        local_alpha=local_alpha,
        local_Re=local_Re,
        local_C_mu=local_C_mu,
        residual_CL=_get(result, "residual_CL", _get(result, "residuals")),
        residual_CM=_get(result, "residual_CM"),
        convergence_history=_get(result, "convergence_history"),
        max_residual=_get(result, "max_residual"),
        iteration_count=_get(result, "iteration_count"),
        converged=_get(result, "converged"),
        failure_reason=_get(result, "failure_reason"),
    )


def add_afc_3d_postprocessing(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds standard AFC 3D post-processing fields to an analysis result dictionary in-place.
    """
    post = postprocess_afc_3d_result(result)
    result["afc_post"] = post
    result["root_bending_moment"] = post.root_bending_moment
    result["spanwise_load"] = post.spanwise_load
    result["local_stall_indicator"] = post.local_stall_indicator
    result["afc_power_proxy"] = post.afc_power_proxy
    if "section_CL" in result:
        result.setdefault("section_cl", result["section_CL"])
    if "section_CD" in result:
        result.setdefault("section_cd", result["section_CD"])
    if "section_CM" in result:
        result.setdefault("section_cm", result["section_CM"])
    result.setdefault("CM", result.get("Cm", None))
    result.setdefault("CY", None)
    result.setdefault("Cl", None)
    result.setdefault("Cn", None)
    result.setdefault("residual_CL", post.residual_CL)
    result.setdefault("residual_CM", post.residual_CM)
    result.setdefault("convergence_history", post.convergence_history)
    result.setdefault("max_residual", post.max_residual)
    result.setdefault("iteration_count", post.iteration_count)
    result.setdefault("converged", post.converged)
    result.setdefault("failure_reason", post.failure_reason)
    result["postprocess_schema"] = post.to_dict()
    return result


def plot_spanwise_results(
    result: Union[Dict[str, Any], AFC3DPostProcessResult],
    *,
    title: str = "AFC 3D Spanwise Results",
    show: bool = True,
    savefig: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any]:
    """
    Plots spanwise load, section coefficients, confidence/stall indicators, and residuals.
    """
    import matplotlib.pyplot as plt

    post = (
        result
        if isinstance(result, AFC3DPostProcessResult)
        else postprocess_afc_3d_result(result)
    )

    y = post.spanwise_y
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharex=True)
    axes = axes.flatten()

    if post.spanwise_load is not None:
        axes[0].plot(y, post.spanwise_load, "o-", label="spanwise load")
    axes[0].set_ylabel("Load [N/m]")
    axes[0].grid(True, alpha=0.25)

    if post.local_C_mu is not None:
        ax2 = axes[0].twinx()
        ax2.plot(y, post.local_C_mu, "s--", color="tab:orange", label="C_mu")
        ax2.set_ylabel("C_mu")

    if post.section_CL is not None:
        axes[1].plot(y, post.section_CL, "o-", label="section CL")
    if post.section_CD is not None:
        axes[1].plot(y, post.section_CD, "s-", label="section CD")
    if post.section_CM is not None:
        axes[1].plot(y, post.section_CM, "^-", label="section CM")
    axes[1].set_ylabel("Section coeff.")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    if post.analysis_confidence is not None:
        axes[2].plot(y, post.analysis_confidence, "o-", label="confidence")
    if post.local_stall_indicator is not None:
        axes[2].plot(y, post.local_stall_indicator, "s-", label="stall proxy")
    axes[2].set_xlabel("y [m]")
    axes[2].set_ylabel("0..1")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    plotted_residual = False
    if post.residual_CL is not None:
        axes[3].plot(y, post.residual_CL, "o-", label="residual CL")
        plotted_residual = True
    if post.residual_CM is not None:
        axes[3].plot(y, post.residual_CM, "s-", label="residual CM")
        plotted_residual = True
    if plotted_residual:
        axes[3].axhline(0, color="k", linewidth=0.8)
        axes[3].legend()
    axes[3].set_xlabel("y [m]")
    axes[3].set_ylabel("Residual")
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()

    if savefig is not None:
        savefig = Path(savefig)
        savefig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
