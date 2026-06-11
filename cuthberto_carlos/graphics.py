"""Graphics for match prediction outputs."""

import os
import tempfile

from jax import numpy as jnp
import pandas as pd
import plotnine as pn
from PIL import Image


def _date_str(date) -> str:
    return pd.to_datetime(date).strftime("%Y-%m-%d")


def make_graphic(
    match_data: pd.Series,
    skills_mean,
    skills_cov,
    probs_grid,
    probs_results,
    team_name_to_colors: dict[str, tuple[str, str] | list[str]],
    save_path: str,
    max_goals_grid: int = 4,
) -> None:
    """Make a graphic showing the predicted probabilities for the match.

    Args:
        match_data: A pandas Series containing the data for the match.
        skills_mean: A 2D array of shape (2, 2) containing the mean attack and defence
            skills for the home and away teams.
        skills_cov: A 3D array of shape (2, 2, 2) containing the covariance of the
            attack and defence skills for the home and away teams.
        probs_grid: A 2D array of shape (max_goals+1, max_goals+1) containing the
            predicted probabilities for each scoreline.
        probs_results: A 1D array of shape (3,) containing the predicted probabilities
            for each result (draw, home win, away win).
        team_name_to_colors: Mapping from team names to primary and secondary colors.
        save_path: The path to save the graphic to.
        max_goals_grid: The maximum scoreline to show on the score grid.
    """
    home_team_str = match_data["home_team"]
    away_team_str = match_data["away_team"]
    tournament = match_data["tournament"]
    city = match_data["city"]
    date = _date_str(match_data["date"])

    home_attack_defence_mean = skills_mean[0]
    home_attack_defence_sd = jnp.diag(skills_cov[0]) ** 0.5
    away_attack_defence_mean = skills_mean[1]
    away_attack_defence_sd = jnp.diag(skills_cov[1]) ** 0.5
    strength_data = pd.DataFrame(
        [
            {
                "team": home_team_str,
                "metric": "Attack",
                "mean": float(home_attack_defence_mean[0]),
                "sd": float(home_attack_defence_sd[0]),
                "color": "home",
            },
            {
                "team": home_team_str,
                "metric": "Defence",
                "mean": float(home_attack_defence_mean[1]),
                "sd": float(home_attack_defence_sd[1]),
                "color": "home",
            },
            {
                "team": away_team_str,
                "metric": "Attack",
                "mean": float(away_attack_defence_mean[0]),
                "sd": float(away_attack_defence_sd[0]),
                "color": "away",
            },
            {
                "team": away_team_str,
                "metric": "Defence",
                "mean": float(away_attack_defence_mean[1]),
                "sd": float(away_attack_defence_sd[1]),
                "color": "away",
            },
        ]
    )
    strength_data["lower"] = strength_data["mean"] - strength_data["sd"]
    strength_data["upper"] = strength_data["mean"] + strength_data["sd"]
    strength_data["label"] = strength_data["team"] + " " + strength_data["metric"]
    strength_data["label"] = pd.Categorical(
        strength_data["label"],
        categories=[
            f"{away_team_str} Defence",
            f"{away_team_str} Attack",
            f"{home_team_str} Defence",
            f"{home_team_str} Attack",
        ],
        ordered=True,
    )

    home_team_color, _ = team_name_to_colors.get(home_team_str, ("#888888", "#BBBBBB"))
    away_team_primary_color, away_team_secondary_color = team_name_to_colors.get(
        away_team_str, ("#888888", "#BBBBBB")
    )
    away_team_color = (
        away_team_primary_color
        if away_team_primary_color != home_team_color
        else away_team_secondary_color
    )

    score_probs = jnp.asarray(probs_grid[: max_goals_grid + 1, : max_goals_grid + 1])
    score_data = pd.DataFrame(
        [
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "prob": float(score_probs[home_goals, away_goals]),
                "label": f"{100 * float(score_probs[home_goals, away_goals]):.0f}%",
            }
            for home_goals in range(max_goals_grid + 1)
            for away_goals in range(max_goals_grid + 1)
        ]
    )

    result_data = pd.DataFrame(
        [
            {"result": "Draw", "prob": float(probs_results[0]), "color": "draw"},
            {
                "result": f"{home_team_str} win",
                "prob": float(probs_results[1]),
                "color": "home",
            },
            {
                "result": f"{away_team_str} win",
                "prob": float(probs_results[2]),
                "color": "away",
            },
        ]
    )
    result_data["result"] = pd.Categorical(
        result_data["result"],
        categories=[
            f"{away_team_str} win",
            "Draw",
            f"{home_team_str} win",
        ],
        ordered=True,
    )
    result_data["label"] = result_data["prob"].map(lambda x: f"{100 * x:.1f}%")

    header_title = pd.DataFrame(
        [{"x": 0, "y": 1.4, "label": f"{home_team_str} vs {away_team_str}"}]
    )
    header_subtitle = pd.DataFrame(
        [
            {
                "x": 0,
                "y": 0.35,
                "label": (
                    f"{date}  |  {tournament}  |  {city}, {match_data['country']}"
                ),
            }
        ]
    )
    header_plot = (
        pn.ggplot()
        + pn.geom_text(
            mapping=pn.aes("x", "y", label="label"),
            data=header_title,
            ha="left",
            va="center",
            size=36,
            fontweight="bold",
            color="#15181D",
            show_legend=False,
        )
        + pn.geom_text(
            mapping=pn.aes("x", "y", label="label"),
            data=header_subtitle,
            ha="left",
            va="center",
            size=18,
            color="#414853",
            show_legend=False,
        )
        + pn.coord_cartesian(xlim=(0, 12), ylim=(-0.2, 1.85))
        + pn.theme_void()
        + pn.theme(figure_size=(13.5, 1.55))
    )

    strength_limit = 1.5
    strength_plot = (
        pn.ggplot(strength_data, pn.aes("mean", "label", color="color"))
        + pn.geom_vline(xintercept=0, color="#C9CED6", size=0.6)
        + pn.geom_segment(
            pn.aes(x="lower", xend="upper", y="label", yend="label"),
            size=1.7,
            alpha=0.8,
        )
        + pn.geom_point(size=4.2)
        + pn.scale_color_manual(
            values={"home": home_team_color, "away": away_team_color},
        )
        + pn.scale_x_continuous(limits=(-strength_limit, strength_limit))
        + pn.labs(x="", y="", title="Team strengths")
        + pn.theme_minimal()
        + pn.theme(
            figure_size=(7.2, 3.1),
            plot_title=pn.element_text(size=17, weight="bold"),
            axis_text=pn.element_text(size=12),
            axis_text_y=pn.element_text(size=12),
            legend_position="none",
        )
    )

    score_plot = (
        pn.ggplot(score_data, pn.aes("home_goals", "away_goals", fill="prob"))
        + pn.geom_tile(color="white", size=0.8)
        + pn.geom_text(pn.aes(label="label"), size=11, color="#20242A")
        + pn.scale_x_continuous(
            breaks=list(range(max_goals_grid + 1)),
            limits=(-0.5, max_goals_grid + 0.5),
        )
        + pn.scale_y_continuous(
            breaks=list(range(max_goals_grid + 1)),
            limits=(-0.5, max_goals_grid + 0.5),
        )
        + pn.scale_fill_gradient(low="#F3F0E8", high="#3F8F64")
        + pn.coord_fixed()
        + pn.labs(
            x=f"{home_team_str} goals",
            y=f"{away_team_str} goals",
            fill="Probability",
        )
        + pn.theme_minimal()
        + pn.theme(
            figure_size=(8.0, 8.0),
            plot_title=pn.element_text(size=17, weight="bold"),
            axis_title=pn.element_text(size=16),
            axis_text=pn.element_text(size=13),
            legend_position="none",
        )
    )

    result_plot = (
        pn.ggplot(result_data, pn.aes("result", "prob", fill="color"))
        + pn.geom_col(width=0.62)
        + pn.geom_text(
            pn.aes(label="label"),
            nudge_y=0.035,
            ha="left",
            size=12,
            color="#20242A",
        )
        + pn.scale_fill_manual(
            values={
                "home": home_team_color,
                "away": away_team_color,
                "draw": "#7A808A",
            },
        )
        + pn.scale_y_continuous(
            limits=(0, 1),
            labels=lambda values: [f"{100 * v:.0f}%" for v in values],
        )
        + pn.coord_flip()
        + pn.labs(x="", y="", title="Result probabilities")
        + pn.theme_minimal()
        + pn.theme(
            figure_size=(7.2, 2.5),
            plot_title=pn.element_text(size=17, weight="bold"),
            axis_text=pn.element_text(size=12),
            axis_text_y=pn.element_text(size=12),
            legend_position="none",
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        header_path = os.path.join(tmp_dir, "header.png")
        strength_path = os.path.join(tmp_dir, "strength.png")
        score_path = os.path.join(tmp_dir, "score.png")
        result_path = os.path.join(tmp_dir, "result.png")
        header_plot.save(header_path, width=13.5, height=1.55, dpi=160, verbose=False)
        strength_plot.save(strength_path, width=7.2, height=3.1, dpi=160, verbose=False)
        score_plot.save(score_path, width=8.0, height=8.0, dpi=160, verbose=False)
        result_plot.save(result_path, width=7.2, height=2.5, dpi=160, verbose=False)

        header_img = Image.open(header_path).convert("RGB")
        strength_img = Image.open(strength_path).convert("RGB")
        score_img = Image.open(score_path).convert("RGB")
        result_img = Image.open(result_path).convert("RGB")

        padding = 28
        gap = 46
        left_column_gap = 300
        content_height = max(
            strength_img.height + left_column_gap + result_img.height,
            score_img.height,
        )
        canvas_width = max(
            header_img.width,
            padding * 2 + strength_img.width + gap + score_img.width,
        )
        canvas_height = padding + header_img.height + gap + content_height + padding
        canvas = Image.new("RGB", (canvas_width, canvas_height), "#FFFFFF")
        canvas.paste(header_img, (padding, padding))
        left_x = padding
        top_y = padding + header_img.height + gap
        right_x = padding + strength_img.width + gap
        canvas.paste(strength_img, (left_x, top_y))
        canvas.paste(
            result_img, (left_x, top_y + strength_img.height + left_column_gap)
        )
        canvas.paste(score_img, (right_x, top_y))

        canvas.save(save_path)
