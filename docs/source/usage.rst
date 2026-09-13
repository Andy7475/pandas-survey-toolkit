Usage
=====

Basic Examples
--------------

Here's how to use the main features of the toolkit:

.. code-block:: python

    import pandas as pd
    from pandas_survey_toolkit import analytics, nlp, vis

    # Load your data (with free text comments)
    df = pd.read_csv('survey_data.csv')

    # Preprocess text
    df = df.preprocess_text(input_column='comments')

    # Extract sentiment
    df = df.extract_sentiment(input_column="comments")
    
    # Extract keywords
    df = df.extract_keywords(input_column='comments')

    # Cluster comments
    df = df.cluster_comments(input_column='comments')

Clustering a Likert Survey (Cosine Distance)
---------------------------------------------

For strongly-disagree ... neutral ... strongly-agree style questions, the
toolkit clusters both respondents *and* questions straight from the encoded
answers (no NLP on the question text). Responses are encoded on a 3-point
scale (-1 / 0 / +1) and grouped using **cosine distance**, which compares the
*direction* of a response vector rather than raw magnitude - so respondents
who agree with everything and respondents who disagree with everything
separate cleanly, and a genuine agree/disagree clash counts as more distant
than simple indifference.

.. code-block:: python

    import pandas_survey_toolkit.nlp  # registers the .cluster_* methods
    from pandas_survey_toolkit.vis import (
        cluster_heatmap_plot,
        survey_clustermap,
        plot_respondent_dendrogram,
    )

    questions = ["q1_ease_of_use", "q2_customer_service", "q3_value_for_money"]

    # Cluster both respondents and questions in one call, and stash the
    # orderings the plotting helpers need.
    survey = df.cluster_survey(columns=questions)

    # Heatmap: sentiment per respondent-cluster x question (best for many
    # respondents / few clusters).
    encoded = [f"likert_encoded_{q}" for q in questions]
    cluster_heatmap_plot(
        survey, respondent_col="respondent_cluster_id", question_cols=encoded
    )

    # Clustermap: every respondent x question, with marginal dendrograms
    # (best for small/medium surveys).
    survey_clustermap(df, columns=questions, label_col="respondent_id")

    # Inspect the cosine dendrogram behind the respondent clustering.
    df_cos = df.cluster_respondents_cosine(columns=questions)
    plot_respondent_dendrogram(df_cos, label_col="respondent_id")

``cluster_respondents(method="auto")`` (used internally by ``cluster_survey``)
picks ``cluster_respondents_cosine`` for small/medium surveys and switches to
``cluster_respondents_umap`` (UMAP + HDBSCAN) once the respondent count grows
past ``size_threshold`` (default 1000), since the cosine engine is
``O(n^2)``. See :doc:`clustering_methods_comparison` for the maths behind the
distance, and how to pick a ``distance_threshold``.

.. toctree::
   :maxdepth: 1
   :hidden: