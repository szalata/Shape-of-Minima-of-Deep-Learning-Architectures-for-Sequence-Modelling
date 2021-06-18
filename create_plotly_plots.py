import os
import numpy as np
import plotly.graph_objects as go


output_dir = os.path.join("output", "seq_cls", "fixed_length")
plotly_dir = "output"


subdirs = sorted([x[0] for x in os.walk(output_dir)][1:])
subdir_names = [subdir.split("/")[-1] for subdir in subdirs if os.path.exists(os.path.join(subdir, "test_loss.npy"))]
test_losses = [np.load(os.path.join(subdir, "test_loss.npy")) for subdir in subdirs if os.path.exists(os.path.join(subdir, "test_loss.npy"))]
test_losses = np.array(test_losses)
hessian_test = [np.load(os.path.join(subdir, "hessian_test.npy")) for subdir in subdirs if os.path.exists(os.path.join(subdir, "hessian_test.npy"))]
hessian_test = np.array(hessian_test)
# remove GRU
hessian_subdir_names = subdir_names[len(subdir_names)//3:]

def plotly_plot_losses(y, text, show_labels=False, sort=False, remove_smallest=0, remove_largest=0, fig_name="fig", plot_title="", yaxis_title=""):
    x = np.tile(np.arange(len(y) // 3), 3)
    
    if sort:
        text = [label for _, label in sorted(zip(y, text))]
        y = sorted(y)
    
    mode = "markers"
    if show_labels:
        mode += "+text"
    
    if remove_largest == 0:
        y_narrowed = y[remove_smallest:]
        text_narrowed = text[remove_smallest:]
    else:
        y_narrowed = y[remove_smallest:-remove_largest]
        text_narrowed = text[remove_smallest:-remove_largest]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
                mode=mode,
                x=x[:len(x) // 3],
                y=y_narrowed[:len(y) // 3],
                marker=dict(
                    color='Brown',
                    size=10,
                ),
                text=text[:len(x) // 3],
                showlegend=True,
                name="GRU"
            )
    )
    fig.add_trace(
        go.Scatter(
                mode=mode,
                x=x[len(x) // 3:len(x) // 3 * 2],
                y=y_narrowed[len(y) // 3:len(x) // 3 * 2],
                marker=dict(
                    color='LightSkyBlue',
                    size=10,
                ),
                text=text[len(x) // 3:len(x) // 3 * 2],
                showlegend=True,
                name="LSTM"
            )
    )
    fig.add_trace(
        go.Scatter(
                mode=mode,
                x=x[len(x) // 3 * 2:],
                y=y_narrowed[len(x) // 3 * 2:],
                marker=dict(
                    color='Green',
                    size=10,
                ),
                text=text[len(x) // 3 * 2:],
                showlegend=True,
                name="Transformer"
            )
    )
    fig.update_layout(
        title=plot_title,
        yaxis_title=yaxis_title,
        xaxis = go.XAxis(
            title = 'Model configurations',
            showticklabels=False),
    )
    fig.write_image(os.path.join(plotly_dir, f"{fig_name}.png"))
    
def plotly_plot_hessian(y, text, show_labels=False, sort=False, remove_smallest=0, remove_largest=0, fig_name="fig", plot_title="", yaxis_title=""):
    x = np.tile(np.arange(len(y) // 2), 2)
    
    if sort:
        text = [label for _, label in sorted(zip(y, text))]
        y = sorted(y)
    
    mode = "markers"
    if show_labels:
        mode += "+text"
    
    if remove_largest == 0:
        y_narrowed = y[remove_smallest:]
        text_narrowed = text[remove_smallest:]
    else:
        y_narrowed = y[remove_smallest:-remove_largest]
        text_narrowed = text[remove_smallest:-remove_largest]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
                mode=mode,
                x=x[:len(x) // 2],
                y=y_narrowed[:len(y) // 2],
                marker=dict(
                    color='LightSkyBlue',
                    size=10,
                ),
                text=text[:len(x) // 2],
                showlegend=True,
                name="LSTM"
            )
    )
    fig.add_trace(
        go.Scatter(
                mode=mode,
                x=x[len(x) // 2:],
                y=y_narrowed[len(y) // 2:],
                marker=dict(
                    color='Green',
                    size=10,
                ),
                text=text[:len(x) // 2],
                showlegend=True,
                name="Transformer"
            )
    )
    fig.update_yaxes(range=[0, 6])
    fig.update_xaxes(range=[-1, 27])
    fig.update_yaxes(type="log")
    fig.update_layout(
        title=plot_title,
        yaxis_title=yaxis_title,
        xaxis = go.XAxis(
            title = 'Model configurations',
            showticklabels=False),
    )
    fig.write_image(os.path.join(plotly_dir, f"{fig_name}.png"))
    
plotly_plot_hessian(hessian_test[:, 0], hessian_subdir_names, fig_name="top_eigenvalue", plot_title=r"$ \text{Top }\lambda \text{, LSTM vs Transformer on sequence classification}$", yaxis_title=r"$|\text{top }\lambda|\text{, log scale}$")
plotly_plot_hessian(hessian_test[:, 1], hessian_subdir_names, fig_name="trace", plot_title="Hessian trace , LSTM vs Transformer on sequence classification", yaxis_title=r"Hessian trace, log scale")
plotly_plot_losses(test_losses, subdir_names, fig_name="losses", plot_title="Test loss, GRU vs LSTM vs Transformer on sequence classification", yaxis_title="Test loss")