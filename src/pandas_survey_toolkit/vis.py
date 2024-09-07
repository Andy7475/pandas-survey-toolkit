import altair as alt
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.preprocessing import MinMaxScaler
from pyvis.network import Network
from scipy import stats
import matplotlib.colors as mcolors


def create_keyword_graph(df: pd.DataFrame, keyword_column: str) -> nx.DiGraph:
    """
    Create a NetworkX DiGraph from a DataFrame column containing lists of keywords.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    keyword_column (str): The name of the column containing keyword lists

    Returns:
    nx.DiGraph: A directed graph representing keyword connections
    """
    G = nx.DiGraph()

    for keywords in df[keyword_column]:
        if not isinstance(keywords, list) or len(keywords) == 0:
            continue

        # populate nodes first
        for i, keyword in enumerate(keywords):

            keyword = keyword.encode("utf-8").decode("utf-8")
            # Add node or update node count
            if G.has_node(keyword):
                G.nodes[keyword]["node_count"] += 1
            else:
                G.add_node(keyword, node_count=1, title=keyword)
        # now popualte edges
        if len(keywords) < 2:
            continue
        for i, keyword in enumerate(keywords):
            # Add edge to next keyword if it exists
            keyword = keyword.encode("utf-8").decode("utf-8")

            if i < len(keywords) - 1:
                next_keyword = keywords[i + 1]
                next_keyword = next_keyword.encode("utf-8").decode("utf-8")
                if G.has_edge(keyword, next_keyword):
                    G[keyword][next_keyword]["edge_count"] += 1
                else:
                    G.add_edge(keyword, next_keyword, edge_count=1)

    return G


def visualize_keyword_graph(
    graph: nx.DiGraph,
    output_file: str = None,
    min_edge_count: int = 4,
    min_node_count: int = 4,
):
    """
    Visualize a filtered NetworkX DiGraph using PyViz Network.

    Parameters:
    graph (nx.DiGraph): The input graph to visualize
    output_file (str): The name of the output HTML file (default: None)
    notebook (bool): Whether to display the graph inline in a Jupyter notebook (default: True)
    min_edge_count (int): Minimum edge count to include in the visualization (default: 5)
    min_node_count (int): Minimum node count to include in the visualization (default: 8)

    Returns:
    Network: The PyViz Network object for further customization if needed
    """
    # Filter the graph based on min_edge_count and min_node_count
    filtered_graph = nx.DiGraph()

    for node, data in graph.nodes(data=True):
        if data["node_count"] >= min_node_count:
            filtered_graph.add_node(node, **data)

    for source, target, data in graph.edges(data=True):
        if (
            data["edge_count"] >= min_edge_count
            and source in filtered_graph.nodes
            and target in filtered_graph.nodes
        ):
            filtered_graph.add_edge(source, target, **data)

    # Remove isolated nodes (nodes with no edges)
    filtered_graph.remove_nodes_from(list(nx.isolates(filtered_graph)))

    # Create a PyViz Network object
    net = Network(directed=True, width="100%", height="800px")

    # Add nodes to the network
    for node, data in filtered_graph.nodes(data=True):
        net.add_node(
            node,
            label=data["title"],
            title=f"Keyword: {data['title']}\nCount: {data['node_count']}",
            size=data["node_count"] * 1,
        )  # Adjust the multiplier as needed

    # Add edges to the network
    for source, target, data in filtered_graph.edges(data=True):
        net.add_edge(
            source,
            target,
            title=f"Count: {data['edge_count']}",
            width=data["edge_count"],
        )  # Edge thickness based on count

    # Set some display options
    net.set_options(
        """
    var options = {
      "edges": {
        "arrows": {
          "to": {
            "enabled": true
          }
        },
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "minVelocity": 0.1
      }
    }
    """
    )

    # Save the graph as an interactive HTML file if output_file is provided
    if output_file:
        net.save_graph(output_file)
        print(f"Graph saved to {output_file}")

    # Display the graph inline if in a notebook
    else:
        net.save_graph("keyword_graph.html")


def dense_rank(series):
    """
    Compute dense rank for a series.
    This will assign the same rank to tied values, but ranks will be continuous.
    """
    return stats.rankdata(series, method="dense")


def create_keyword_sentiment_df_simple(df):
    sentiment_counts = df["sentiment"].value_counts()
    total_positive = sentiment_counts.get("positive", 0)
    total_negative = sentiment_counts.get("negative", 0)

    keyword_sentiments = {}

    for _, row in df.iterrows():
        sentiment = row["sentiment"]
        if sentiment == "neutral":
            continue
        for word in row["keywords"]:
            if word not in keyword_sentiments:
                keyword_sentiments[word] = {"positive": 0, "negative": 0}
            keyword_sentiments[word][sentiment] += 1

    result_df = pd.DataFrame(
        [
            {
                "word": word,
                "sentiment_score": (
                    counts["positive"] / total_positive if total_positive else 0
                )
                - (counts["negative"] / total_negative if total_negative else 0),
            }
            for word, counts in keyword_sentiments.items()
        ]
    )

    return result_df


def create_keyword_sentiment_df(df):
    sentiment_counts = df["sentiment"].value_counts()
    total_positive = sentiment_counts.get("positive", 0)
    total_negative = sentiment_counts.get("negative", 0)

    positive_keywords = {}
    negative_keywords = {}

    for _, row in df.iterrows():
        if row["sentiment"] == "positive":
            for word in row["keywords"]:
                positive_keywords[word] = positive_keywords.get(word, 0) + 1
        elif row["sentiment"] == "negative":
            for word in row["keywords"]:
                negative_keywords[word] = negative_keywords.get(word, 0) + 1

    all_keywords = set(positive_keywords.keys()) | set(negative_keywords.keys())

    result_df = pd.DataFrame(
        {
            "word": list(all_keywords),
            "sentiment_positive": [
                positive_keywords.get(word, 0) / total_positive if total_positive else 0
                for word in all_keywords
            ],
            "sentiment_negative": [
                negative_keywords.get(word, 0) / total_negative if total_negative else 0
                for word in all_keywords
            ],
        }
    )

    # Apply dense ranking
    result_df["sentiment_positive_rank"] = dense_rank(result_df["sentiment_positive"])
    result_df["sentiment_negative_rank"] = dense_rank(result_df["sentiment_negative"])

    # Normalize ranks to [0, 1] range
    result_df["sentiment_positive_scaled"] = (
        result_df["sentiment_positive_rank"] - 1
    ) / (result_df["sentiment_positive_rank"].max() - 1)
    result_df["sentiment_negative_scaled"] = (
        result_df["sentiment_negative_rank"] - 1
    ) / (result_df["sentiment_negative_rank"].max() - 1)

    # Add small random jitter to avoid perfect overlaps
    jitter = 0.01
    result_df["sentiment_positive_jittered"] = result_df[
        "sentiment_positive_scaled"
    ] + np.random.uniform(-jitter, jitter, len(result_df))
    result_df["sentiment_negative_jittered"] = result_df[
        "sentiment_negative_scaled"
    ] + np.random.uniform(-jitter, jitter, len(result_df))

    # Add color coding
    result_df["color"] = np.where(
        result_df["sentiment_positive_jittered"]
        > result_df["sentiment_negative_jittered"],
        "blue",
        "red",
    )

    return result_df


def create_sentiment_color_mapping(sentiment_df):
    """
    Create a dictionary mapping keywords to normalized sentiment scores.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_scores = scaler.fit_transform(sentiment_df[["sentiment_score"]])
    return dict(zip(sentiment_df["word"], normalized_scores.flatten()))


def create_keyword_graph(
    df: pd.DataFrame, keyword_column: str, node_color_mapping: dict = None
) -> nx.DiGraph:
    """
    Create a NetworkX DiGraph from a DataFrame column containing lists of keywords.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    keyword_column (str): The name of the column containing keyword lists
    node_color_mapping (dict): Optional dictionary mapping keywords to color values

    Returns:
    nx.DiGraph: A directed graph representing keyword connections
    """
    G = nx.DiGraph()

    for keywords in df[keyword_column]:
        if not isinstance(keywords, list) or len(keywords) == 0:
            continue

        for i, keyword in enumerate(keywords):
            keyword = keyword.encode("utf-8").decode("utf-8")

            if G.has_node(keyword):
                G.nodes[keyword]["node_count"] += 1
            else:
                G.add_node(keyword, node_count=1, title=keyword, group="keyword")
                if node_color_mapping and keyword in node_color_mapping:
                    G.nodes[keyword]["color_value"] = node_color_mapping[keyword]

        if len(keywords) < 2:
            continue
        for i, keyword in enumerate(keywords):
            keyword = keyword.encode("utf-8").decode("utf-8")
            if i < len(keywords) - 1:
                next_keyword = keywords[i + 1].encode("utf-8").decode("utf-8")
                if G.has_edge(keyword, next_keyword):
                    G[keyword][next_keyword]["edge_count"] += 1
                else:
                    G.add_edge(keyword, next_keyword, edge_count=1)

    return G


def visualize_keyword_graph(
    graph: nx.DiGraph,
    output_file: str = None,
    min_edge_count: int = 4,
    min_node_count: int = 4,
    colormap: str = "RdYlBu",
):
    """
    Visualize a filtered NetworkX DiGraph using PyViz Network.

    Parameters:
    graph (nx.DiGraph): The input graph to visualize
    output_file (str): The name of the output HTML file (default: None)
    min_edge_count (int): Minimum edge count to include in the visualization (default: 4)
    min_node_count (int): Minimum node count to include in the visualization (default: 4)
    colormap (str): Name of the matplotlib colormap to use (default: 'RdYlBu')
    """
    # Filter nodes based on min_node_count
    nodes_to_keep = [
        node
        for node, data in graph.nodes(data=True)
        if data["node_count"] >= min_node_count
    ]
    filtered_graph = graph.subgraph(nodes_to_keep).copy()

    # Filter edges based on min_edge_count
    edges_to_remove = [
        (u, v)
        for u, v, data in filtered_graph.edges(data=True)
        if data["edge_count"] < min_edge_count
    ]
    filtered_graph.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    filtered_graph.remove_nodes_from(list(nx.isolates(filtered_graph)))

    net = Network(directed=True, width="100%", height="800px")

    cmap = plt.get_cmap(colormap)

    for node, data in filtered_graph.nodes(data=True):
        if "color_value" in data:
            # Map the color_value from [-1, 1] to [0, 1] for the colormap
            color_val = (data["color_value"] + 1) / 2
            node_color = mcolors.rgb2hex(cmap(color_val))
        else:
            node_color = None

        net.add_node(
            node,
            label=data["title"],
            title=f"Keyword: {data['title']}\nCount: {data['node_count']}\nSentiment: {data.get('color_value', 0):.2f}",
            size=data["node_count"] * 1,
            color=node_color,
        )

    for source, target, data in filtered_graph.edges(data=True):
        net.add_edge(
            source,
            target,
            title=f"Count: {data['edge_count']}",
            width=data["edge_count"],
        )

    net.set_options(
        """
    var options = {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true
          }
        },
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """
    )

    if output_file:
        net.save_graph(output_file)
        print(f"Graph saved to {output_file}")
    else:
        net.save_graph("keyword_graph.html")


def visualize_keyword_graph_force(
    graph: nx.DiGraph,
    output_file: str = None,
    min_edge_count: int = 4,
    min_node_count: int = 4,
    colormap: str = "RdYlBu",
    canvas_width: int = 1000,
    canvas_height: int = 800,
):
    """
    Visualize a filtered NetworkX DiGraph using PyViz Network with sentiment-based positioning.

    Parameters:
    graph (nx.DiGraph): The input graph to visualize
    output_file (str): The name of the output HTML file (default: None)
    min_edge_count (int): Minimum edge count to include in the visualization (default: 4)
    min_node_count (int): Minimum node count to include in the visualization (default: 4)
    colormap (str): Name of the matplotlib colormap to use (default: 'RdYlBu')
    canvas_width (int): Width of the canvas in pixels (default: 1000)
    canvas_height (int): Height of the canvas in pixels (default: 800)
    """
    # Filter nodes based on min_node_count
    nodes_to_keep = [
        node
        for node, data in graph.nodes(data=True)
        if data["node_count"] >= min_node_count
    ]
    filtered_graph = graph.subgraph(nodes_to_keep).copy()

    # Filter edges based on min_edge_count
    edges_to_remove = [
        (u, v)
        for u, v, data in filtered_graph.edges(data=True)
        if data["edge_count"] < min_edge_count
    ]
    filtered_graph.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    filtered_graph.remove_nodes_from(list(nx.isolates(filtered_graph)))

    net = Network(directed=True, width=f"{canvas_width}px", height=f"{canvas_height}px")

    cmap = plt.get_cmap(colormap)

    for node, data in filtered_graph.nodes(data=True):
        if "color_value" in data:
            # Map the color_value from [-1, 1] to [0, 1] for the colormap
            color_val = (data["color_value"] + 1) / 2
            node_color = mcolors.rgb2hex(cmap(color_val))

            # Set x position based on sentiment (color_value)
            x_pos = int((data["color_value"] + 1) * canvas_width / 2)
        else:
            node_color = None
            x_pos = canvas_width // 2  # Neutral position for nodes without sentiment

        net.add_node(
            node,
            label=data["title"],
            title=f"Keyword: {data['title']}\nCount: {data['node_count']}\nSentiment: {data.get('color_value', 0):.2f}",
            size=data["node_count"] * 1,
            color=node_color,
            x=x_pos,
            y=None,
        )  # Let y be determined by the physics engine

    for source, target, data in filtered_graph.edges(data=True):
        net.add_edge(
            source,
            target,
            title=f"Count: {data['edge_count']}",
            width=data["edge_count"],
        )

    net.set_options(
        f"""
    var options = {{
      "nodes": {{
        "font": {{
          "size": 12
        }}
      }},
      "edges": {{
        "arrows": {{
          "to": {{
            "enabled": true
          }}
        }},
        "color": {{
          "inherit": true
        }},
        "smooth": false
      }},
      "physics": {{
        "barnesHut": {{
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95
        }},
        "minVelocity": 0.75
      }},
      "layout": {{
        "randomSeed": 42
      }}
    }}
    """
    )

    if output_file:
        net.save_graph(output_file)
        print(f"Graph saved to {output_file}")
    else:
        net.save_graph("keyword_graph.html")


def plot_word_sentiment(df):
    # Create the scatter plot
    scatter = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "sentiment_positive_jittered:Q",
                title="Positive Sentiment (Dense Rank)",
                scale=alt.Scale(domain=[-0.05, 1.05]),
            ),
            y=alt.Y(
                "sentiment_negative_jittered:Q",
                title="Negative Sentiment (Dense Rank)",
                scale=alt.Scale(domain=[-0.05, 1.05]),
            ),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["blue", "red"], range=["blue", "red"]),
            ),
            tooltip=["word", "sentiment_positive", "sentiment_negative"],
        )
    )

    # Create the text labels
    text = scatter.mark_text(align="left", baseline="middle", dx=7).encode(text="word")

    # Create the y=x reference line
    line = (
        alt.Chart(pd.DataFrame({"x": [0, 1]}))
        .mark_line(color="green", strokeDash=[4, 4])
        .encode(x="x", y="x")
    )

    # Combine the scatter plot, text labels, and reference line
    chart = (
        (scatter + text + line)
        .properties(
            width=600, height=600, title="Word Sentiment Analysis (Dense Rank Scaling)"
        )
        .interactive()
    )

    return chart


def plot_pyvis_network(
    nx_graph, output_filename="network.html", height=800, width=1000
):
    """
    Plot a NetworkX graph using PyVis Network with tooltips and variable edge thickness.
    Outputs an HTML file.

    Parameters:
    nx_graph (networkx.Graph): The NetworkX graph to plot
    output_filename (str): The name of the output HTML file
    height (int): Height of the plot in pixels
    width (int): Width of the plot in pixels
    """
    # Create a PyVis network
    pyvis_net = Network(
        height=f"{height}px", width=f"{width}px", directed=nx_graph.is_directed()
    )

    # Add nodes to the PyVis network with tooltips
    for node, node_attrs in nx_graph.nodes(data=True):
        tooltip = node_attrs.get(
            "text", f"Node {node}"
        )  # Use 'text' attribute if available, else use node ID
        size = node_attrs.get("node_count", 1) * 5
        group = node_attrs.get("group", "None")
        pyvis_net.add_node(node, title=tooltip, label=str(node), size=size, group=group)

    # Add edges to the PyVis network with variable width
    for source, target, edge_attrs in nx_graph.edges(data=True):
        width = edge_attrs.get(
            "edge_count", 1
        )  # Use 'edge_width' if available, else default to 1
        pyvis_net.add_edge(source, target, value=width)

    # Set some display options
    pyvis_net.set_options(
        """
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 20
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "springLength": 250,
          "springConstant": 0.001
        },
        "minVelocity": 0.75
      }
    }
    """
    )

    # Save the network as an HTML file
    pyvis_net.save_graph(output_filename)
    print(f"Network plot saved to {output_filename}")


def create_comment_keyword_edges(df, comment_col="comment_id", keyword_col="keywords"):
    """
    Creates an edge list from a DataFrame with comment_id and keywords. Assumes keywords col
    holds lists of keywords

    Parameters:
    df (pandas.DataFrame): DataFrame with 'comment_id' (str) and 'keywords' (list) columns

    Returns:
    networkx.Graph: A graph with edges between comment_ids and their keywords
    """
    edge_list = []
    for _, row in df.iterrows():
        comment_id = row[comment_col]
        keywords = row[keyword_col]

        # Skip if keywords list is empty
        if not keywords:
            continue

        # Add edges between the comment and each of its keywords
        for keyword in keywords:
            # Add the keyword node if it doesn't exist

            # Add an edge between the comment and the keyword
            edge_list.append((comment_id, keyword))

    return edge_list
