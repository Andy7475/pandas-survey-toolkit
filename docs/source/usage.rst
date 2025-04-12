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

.. toctree::
   :maxdepth: 1
   :hidden: