import altair as alt
from scipy import stats
import pandas as pd
import numpy as np

def dense_rank(series):
    """
    Compute dense rank for a series.
    This will assign the same rank to tied values, but ranks will be continuous.
    """
    return stats.rankdata(series, method='dense')

def create_keyword_sentiment_df(df):
    sentiment_counts = df['sentiment'].value_counts()
    total_positive = sentiment_counts.get('positive', 0)
    total_negative = sentiment_counts.get('negative', 0)

    positive_keywords = {}
    negative_keywords = {}

    for _, row in df.iterrows():
        if row['sentiment'] == 'positive':
            for word in row['keywords']:
                positive_keywords[word] = positive_keywords.get(word, 0) + 1
        elif row['sentiment'] == 'negative':
            for word in row['keywords']:
                negative_keywords[word] = negative_keywords.get(word, 0) + 1

    all_keywords = set(positive_keywords.keys()) | set(negative_keywords.keys())

    result_df = pd.DataFrame({
        'word': list(all_keywords),
        'sentiment_positive': [positive_keywords.get(word, 0) / total_positive if total_positive else 0 for word in all_keywords],
        'sentiment_negative': [negative_keywords.get(word, 0) / total_negative if total_negative else 0 for word in all_keywords]
    })

    # Apply dense ranking
    result_df['sentiment_positive_rank'] = dense_rank(result_df['sentiment_positive'])
    result_df['sentiment_negative_rank'] = dense_rank(result_df['sentiment_negative'])

    # Normalize ranks to [0, 1] range
    result_df['sentiment_positive_scaled'] = (result_df['sentiment_positive_rank'] - 1) / (result_df['sentiment_positive_rank'].max() - 1)
    result_df['sentiment_negative_scaled'] = (result_df['sentiment_negative_rank'] - 1) / (result_df['sentiment_negative_rank'].max() - 1)

    # Add small random jitter to avoid perfect overlaps
    jitter = 0.01
    result_df['sentiment_positive_jittered'] = result_df['sentiment_positive_scaled'] + np.random.uniform(-jitter, jitter, len(result_df))
    result_df['sentiment_negative_jittered'] = result_df['sentiment_negative_scaled'] + np.random.uniform(-jitter, jitter, len(result_df))

    # Add color coding
    result_df['color'] = np.where(result_df['sentiment_positive_jittered'] > result_df['sentiment_negative_jittered'], 'blue', 'red')

    return result_df

def plot_word_sentiment(df):
    # Create the scatter plot
    scatter = alt.Chart(df).mark_circle().encode(
        x=alt.X('sentiment_positive_jittered:Q', 
                title='Positive Sentiment (Dense Rank)',
                scale=alt.Scale(domain=[-0.05, 1.05])),
        y=alt.Y('sentiment_negative_jittered:Q', 
                title='Negative Sentiment (Dense Rank)',
                scale=alt.Scale(domain=[-0.05, 1.05])),
        color=alt.Color('color:N', scale=alt.Scale(domain=['blue', 'red'], range=['blue', 'red'])),
        tooltip=['word', 'sentiment_positive', 'sentiment_negative']
    )

    # Create the text labels
    text = scatter.mark_text(align='left', baseline='middle', dx=7).encode(
        text='word'
    )

    # Create the y=x reference line
    line = alt.Chart(pd.DataFrame({'x': [0, 1]})).mark_line(color='green', strokeDash=[4, 4]).encode(
        x='x',
        y='x'
    )

    # Combine the scatter plot, text labels, and reference line
    chart = (scatter + text + line).properties(
        width=600,
        height=600,
        title='Word Sentiment Analysis (Dense Rank Scaling)'
    ).interactive()

    return chart