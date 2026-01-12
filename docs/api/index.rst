API Reference
=============

PEACH provides a scVerse-compatible API with three main modules:

.. autosummary::
   :toctree: generated
   :recursive:

   peach.pp
   peach.tl
   peach.pl

Preprocessing (pp)
------------------

.. autosummary::
   :toctree: generated

   peach.pp.load_data
   peach.pp.generate_synthetic
   peach.pp.prepare_training
   peach.pp.load_pathway_networks
   peach.pp.compute_pathway_scores

Tools (tl)
----------

**Training & Coordinates**

.. autosummary::
   :toctree: generated

   peach.tl.train_archetypal
   peach.tl.hyperparameter_search
   peach.tl.archetypal_coordinates
   peach.tl.assign_archetypes
   peach.tl.extract_archetype_weights
   peach.tl.compute_conditional_centroids
   peach.tl.assign_to_centroids

**Statistical Analysis**

.. autosummary::
   :toctree: generated

   peach.tl.gene_associations
   peach.tl.pathway_associations
   peach.tl.conditional_associations

**Pattern Analysis**

.. autosummary::
   :toctree: generated

   peach.tl.pattern_analysis
   peach.tl.archetype_exclusive_patterns
   peach.tl.specialization_patterns
   peach.tl.tradeoff_patterns

**CellRank Integration**

.. autosummary::
   :toctree: generated

   peach.tl.setup_cellrank
   peach.tl.compute_lineage_pseudotimes
   peach.tl.compute_lineage_drivers
   peach.tl.compute_transition_frequencies
   peach.tl.single_trajectory_analysis

Plotting (pl)
-------------

**Core Visualizations**

.. autosummary::
   :toctree: generated

   peach.pl.archetypal_space
   peach.pl.archetypal_space_multi
   peach.pl.training_metrics
   peach.pl.elbow_curve

**Archetype Analysis**

.. autosummary::
   :toctree: generated

   peach.pl.archetype_positions
   peach.pl.archetype_positions_3d
   peach.pl.archetype_statistics
   peach.pl.dotplot

**Pattern Visualization**

.. autosummary::
   :toctree: generated

   peach.pl.pattern_dotplot
   peach.pl.pattern_summary_barplot
   peach.pl.pattern_heatmap

**Trajectory Analysis**

.. autosummary::
   :toctree: generated

   peach.pl.fate_probabilities
   peach.pl.lineage_drivers

Configuration
-------------

.. autosummary::
   :toctree: generated

   peach.tl.SearchConfig
