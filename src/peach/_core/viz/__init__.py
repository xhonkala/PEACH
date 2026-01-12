# Visualization modules - consolidated and organized

# 3D Archetypal space visualization
# Training and performance visualization
from .training_viz import (
    plot_basic_metrics,
    plot_convergence_analysis,
    plot_cv_fold_consistency,
    plot_cv_training_histories,
    plot_hull_metrics,
    plot_hyperparameter_elbow,
    plot_model_performance,
    plot_training_metrics,
    print_space_stats,
    save_metrics_plot,
)
from .viz_convex import plot_archetype_weights, visualize_archetypal_space, visualize_convex_data
