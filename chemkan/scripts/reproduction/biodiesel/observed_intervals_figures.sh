#!/bin/bash
# Regenerate figures from the completed observed-interval experiment; no training.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
require_python

figure_script plot_biodiesel_interval_objective.py
figure_script fig03_biodiesel_trajectories.py --observed-intervals
figure_script fig05_biodiesel_loss.py --observed-intervals

info "Paper counterparts: results/experiments/biodiesel_observed_intervals/figures/fig3.pdf and fig5b.pdf"
info "The interval objective, species bars, and time/species heatmap are additional biodiesel diagnostics."
