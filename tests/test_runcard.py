"""Step 4: expression compilation, runcard validation, and the pop-dyn CLI.

The load-bearing test here is the equivalence one: a runcard and an equivalent
hand-written system must produce identical numbers, so the declarative and
Python front-ends cannot drift apart.
"""

import jax.numpy as jnp
import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

from popdynamics import System, classify, fixed_points, models
from popdynamics.cli import main, run
from popdynamics.expressions import ExpressionError, compile_rhs
from popdynamics.runcard import PlotSpec, RuncardError, from_dict, load

LV = {
    "system": {
        "variables": ["u", "v"],
        "equations": {"u": "u * (1 - v - mu1 * u)", "v": "alpha * v * (u - 1 - mu2 * v)"},
        "parameters": {"alpha": 1.0, "mu1": 0.1, "mu2": 0.1},
    },
    "domain": {"u": [0.0, 2.6], "v": [0.0, 2.6]},
}


def _card(**overrides):
    data = {k: dict(v) if isinstance(v, dict) else v for k, v in LV.items()}
    data.update(overrides)
    return data


# --- expression compilation ---------------------------------------------


def test_compiled_rhs_matches_hand_written():
    rhs = compile_rhs(("u", "v"), LV["system"]["equations"], LV["system"]["parameters"])
    system = System(rhs=rhs, names=("u", "v"), params=LV["system"]["parameters"])
    reference = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    assert jnp.allclose(system.f([1.2, 0.8]), reference.f([1.2, 0.8]))


def test_expressions_are_differentiable():
    """Autodiff must work through eval'd strings, or the runcard path is crippled."""
    rhs = compile_rhs(("u", "v"), LV["system"]["equations"], LV["system"]["parameters"])
    system = System(rhs=rhs, names=("u", "v"), params=LV["system"]["parameters"])
    reference = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    assert jnp.allclose(system.jacobian([1.2, 0.8]), reference.jacobian([1.2, 0.8]))


def test_expressions_support_vmap():
    rhs = compile_rhs(("u", "v"), {"u": "u * (1 - v)", "v": "v * (u - 1)"}, {})
    system = System(rhs=rhs, names=("u", "v"))
    ys = jnp.array([[1.0, 1.0], [2.0, 0.5]])
    assert system.f_fn_batched(ys, {}).shape == (2, 2)


def test_allowed_functions_are_usable():
    rhs = compile_rhs(("x",), {"x": "-exp(-x) * sqrt(abs(x)) + tanh(x) - pi * 0"}, {})
    assert jnp.isfinite(System(rhs=rhs, names=("x",)).f([1.0])).all()


def test_constant_equation_is_allowed():
    system = System(rhs=compile_rhs(("x", "y"), {"x": "1.0", "y": "-y"}, {}), names=("x", "y"))
    assert jnp.allclose(system.f([3.0, 2.0]), jnp.array([1.0, -2.0]))


def test_parameters_are_read_at_call_time():
    """Values come from params, so with_params works on a compiled runcard system."""
    rhs = compile_rhs(("x",), {"x": "-k * x"}, {"k": 1.0})
    system = System(rhs=rhs, names=("x",), params={"k": 1.0})
    assert float(system.with_params(k=5.0).f([2.0])[0]) == pytest.approx(-10.0)


@pytest.mark.parametrize(
    "equations, message",
    [
        ({"u": "u*(1-v", "v": "v"}, "could not parse"),
        ({"u": "u * mu3", "v": "v"}, "unknown name"),
        ({"u": "u.__class__", "v": "v"}, "Attribute is not allowed"),
        ({"u": "__import__('os')", "v": "v"}, "unknown name"),
        ({"u": "[x for x in (1,2)]", "v": "v"}, "not allowed"),
        ({"u": "(lambda: 1)()", "v": "v"}, "allowed"),
        ({"u": "u"}, "no equation given"),
        ({"u": "u", "v": "v", "w": "w"}, "unknown variable"),
    ],
)
def test_bad_expressions_are_rejected(equations, message):
    with pytest.raises(ExpressionError, match=message):
        compile_rhs(("u", "v"), equations, {"alpha": 1.0})


def test_variable_parameter_name_clash_rejected():
    with pytest.raises(ExpressionError, match="both a variable and a parameter"):
        compile_rhs(("u",), {"u": "u"}, {"u": 1.0})


# --- runcard parsing -----------------------------------------------------


def test_parses_equations_form():
    card = from_dict(_card(name="lv"))
    assert card.name == "lv"
    assert card.system.names == ("u", "v")
    assert card.bounds == [(0.0, 2.6), (0.0, 2.6)]


def test_model_shorthand():
    card = from_dict({
        "system": {"model": "damped_lotka_volterra", "parameters": {"alpha": 2.0, "mu1": 0.2}},
        "domain": {"u": [0, 3], "v": [0, 3]},
    })
    assert card.system.params["alpha"] == 2.0
    assert card.system.params["mu1"] == 0.2
    assert card.system.params["mu2"] == 0.1  # factory default preserved


def test_variable_order_follows_the_variables_key():
    card = from_dict({
        "system": {"variables": ["v", "u"], "equations": {"u": "u", "v": "v"}},
        "domain": {"u": [0, 1], "v": [0, 1]},
    })
    assert card.system.names == ("v", "u")
    assert card.bounds == [(0.0, 1.0), (0.0, 1.0)]


def test_output_defaults_to_pdf_when_unspecified():
    card = from_dict(_card(plots=[{"type": "phase_portrait"}]), source="rc.yaml")
    assert card.plots[0].output == "rc_phase_portrait_0.pdf"


def test_plot_specs_are_parsed():
    card = from_dict(_card(plots=[
        {"type": "phase_portrait", "output": "a.pdf", "nullclines": True},
        {"type": "timeseries", "output": "b.pdf", "initial_conditions": [[1, 1]]},
    ]))
    assert [p.type for p in card.plots] == ["phase_portrait", "timeseries"]
    assert card.plots[0] == PlotSpec("phase_portrait", "a.pdf", {"nullclines": True})


@pytest.mark.parametrize(
    "data, message",
    [
        ({"domain": {}}, "missing required key 'system'"),
        (_card(domain=None), "missing required key 'domain'"),
        ({"bogus": 1, **LV}, "unknown top-level key"),
        ({"system": {"model": "nope"}, "domain": {}}, "unknown model"),
        ({"system": {"model": "lotka_volterra", "parameters": {"beta": 1}},
          "domain": {}}, "has no parameter"),
        ({"system": {"model": "lotka_volterra", "equations": {"u": "u"}},
          "domain": {}}, "not both"),
        ({"system": {"variables": ["u"]}, "domain": {}}, "needs either 'model' or 'equations'"),
        ({"system": {"equations": {"u": "u"}}, "domain": {"v": [0, 1]}}, "no range for variable"),
        ({"system": {"equations": {"u": "u"}}, "domain": {"u": [0, 1], "z": [0, 1]}},
         "unknown variable"),
        ({"system": {"equations": {"u": "u"}}, "domain": {"u": [2, 1]}}, "must be increasing"),
        ({"system": {"equations": {"u": "u"}}, "domain": {"u": [0, 1, 2]}}, r"must be \[lo, hi\]"),
        (_card(plots=[{"output": "a.pdf"}]), "missing 'type'"),
        (_card(plots=[{"type": "bogus"}]), "unknown plot type"),
        (_card(plots="not-a-list"), "'plots' must be a list"),
        (_card(analysis={"bogus": True}), "unknown key.*under 'analysis'"),
        ({"system": {"equations": {"u": "u * qq"}}, "domain": {"u": [0, 1]}}, "unknown name"),
    ],
)
def test_invalid_runcards_name_the_problem(data, message):
    with pytest.raises(RuncardError, match=message):
        from_dict(data, source="rc.yaml")


def test_load_reports_file_problems(tmp_path):
    with pytest.raises(RuncardError, match="could not read"):
        load(tmp_path / "missing.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(RuncardError, match="empty"):
        load(empty)
    bad = tmp_path / "bad.yaml"
    bad.write_text("system: [unclosed\n")
    with pytest.raises(RuncardError, match="invalid YAML"):
        load(bad)


# --- the two front-ends must agree --------------------------------------


def test_runcard_and_python_paths_give_identical_numbers():
    """The whole point of keeping the CLI a thin layer."""
    card = from_dict(_card())
    reference = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)

    from_card = fixed_points(card.system, card.bounds)
    from_python = fixed_points(reference, [(0.0, 2.6), (0.0, 2.6)])
    assert jnp.allclose(from_card, from_python)

    for a, b in zip(from_card, from_python):
        fa, fb = classify(card.system, a), classify(reference, b)
        assert fa.label == fb.label
        assert jnp.allclose(fa.eigenvalues, fb.eigenvalues)


# --- the CLI -------------------------------------------------------------


@pytest.fixture
def runcard_file(tmp_path):
    path = tmp_path / "damped_lv.yaml"
    path.write_text(yaml.safe_dump(_card(
        name="damped_lv",
        analysis={"fixed_points": True},
        plots=[
            {"type": "phase_portrait", "output": "phase.pdf",
             "trajectories": {"initial_conditions": [[2.4, 0.6]], "t_span": [0, 40]}},
            {"type": "timeseries", "output": "time.pdf",
             "initial_conditions": [[2.4, 0.6]], "t_span": [0, 40]},
        ],
    )))
    return path


def test_cli_writes_figures_and_report(runcard_file, tmp_path, capsys):
    out = tmp_path / "out"
    assert main([str(runcard_file), "-o", str(out)]) == 0
    assert (out / "phase.pdf").exists()
    assert (out / "time.pdf").exists()
    assert (out / "damped_lv_report.txt").exists()

    assert (out / "phase.pdf").read_bytes().startswith(b"%PDF")
    report = (out / "damped_lv_report.txt").read_text()
    assert "stable spiral" in report and "saddle" in report
    assert "wrote" in capsys.readouterr().out


def test_cli_defaults_output_to_the_runcard_directory(runcard_file):
    assert main([str(runcard_file)]) == 0
    assert (runcard_file.parent / "phase.pdf").exists()


def test_cli_reports_bad_runcards_without_a_traceback(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"system": {"model": "nope"}, "domain": {}}))
    assert main([str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "unknown model" in err
    assert "Traceback" not in err


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.yaml")]) == 1
    assert "could not read" in capsys.readouterr().err


def test_layers_can_be_switched_off_from_the_runcard(tmp_path):
    card = from_dict(_card(plots=[{
        "type": "phase_portrait", "output": "p.pdf",
        "vector_field": False, "nullclines": False, "fixed_points": False,
    }]))
    assert run(card, tmp_path) == [tmp_path / "p.pdf"]


def test_timeseries_without_initial_conditions_is_an_error(tmp_path):
    card = from_dict(_card(plots=[{"type": "timeseries", "output": "t.pdf"}]))
    with pytest.raises(RuncardError, match="needs 'initial_conditions'"):
        run(card, tmp_path)


def test_report_handles_a_domain_with_no_fixed_points(tmp_path, capsys):
    card = from_dict({
        "system": {"equations": {"u": "1.0", "v": "1.0"}},
        "domain": {"u": [0, 1], "v": [0, 1]},
        "analysis": {"fixed_points": True},
    })
    run(card, tmp_path)
    assert "none found" in capsys.readouterr().out


def test_shipped_runcard_runs(tmp_path):
    card = load("runcards/damped_lv.yaml")
    written = run(card, tmp_path)
    assert len(written) == 3
    assert all(p.exists() for p in written)


# --- one-dimensional systems --------------------------------------------


ALLEE = {
    "system": {
        "variables": ["N"],
        "equations": {"N": "r * N + s * N**2"},
        "parameters": {"r": -1.0, "s": 1.0},
    },
    "domain": {"N": [-0.3, 1.8]},
}


def test_plot_type_must_match_the_system_dimension():
    with pytest.raises(RuncardError, match="needs a 2-variable system"):
        from_dict({**ALLEE, "plots": [{"type": "phase_portrait", "output": "p.pdf"}]},
                  source="rc.yaml")


def test_timeseries_works_in_any_dimension(tmp_path):
    card = from_dict({
        **ALLEE,
        "plots": [{"type": "timeseries", "output": "t.pdf", "initial_conditions": [[0.5]]}],
    })
    assert run(card, tmp_path) == [tmp_path / "t.pdf"]


def test_escaping_trajectory_is_reported_clearly(tmp_path):
    """Above the Allee threshold the population runs away; say so, do not crash."""
    card = from_dict({
        **ALLEE,
        "plots": [{"type": "timeseries", "output": "t.pdf",
                   "initial_conditions": [[1.5]], "t_span": [0, 50]}],
    })
    with pytest.raises(RuncardError, match="could not integrate"):
        run(card, tmp_path)
