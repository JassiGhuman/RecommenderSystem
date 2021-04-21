import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib



# Helper Function to create a heatmap after reading data from .csv file using read_csv function.
"""
Create a heatmap from a numpy array and two lists of labels.

Parameters
----------
data
    A 2D numpy array of shape (N, M).
row_labels
    A list or array of length N with the labels for the rows.
col_labels
    A list or array of length M with the labels for the columns.
ax
    A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
    not provided, use current axes or create a new one.  Optional.
cbar_kw
    A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
cbarlabel
    The label for the colorbar.  Optional.
**kwargs
    All other arguments are forwarded to `imshow`.
"""

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar



# annotate_heatmap(im,data,valfmt,textcolors,threshhold,kwargs)
"""
A function to annotate a heatmap.

Parameters
----------
im
    The AxesImage to be labeled.
data
    Data used to annotate.  If None, the image's data is used.  Optional.
valfmt
    The format of the annotations inside the heatmap.  This should either
    use the string format method, e.g. "$ {x:.2f}", or be a
    `matplotlib.ticker.Formatter`.  Optional.
textcolors
    A pair of colors.  The first is used for values below a threshold,
    the second for those above.  Optional.
threshold
    Value in data units according to which the colors from textcolors are
    applied.  If None (the default) uses the middle of the colormap as
    separation.  Optional.
**kwargs
    All other arguments are forwarded to each call to `text` used to create
    the text labels.
"""
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):


    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

#Function to read data from .csv file, assumes that the user has the dataset directory named "expedia-hotel-recommendations" 
# containing train.csv inside it placed in their django project drectory.
def read_csv(no_of_rows,is_random):
    dataset = "./expedia-hotel-recommendations/train.csv"
    if is_random == True:
        dataFrame = pd.read_csv(dataset,nrows = no_of_rows, random_state = 100)
    else:
        dataFrame = pd.read_csv(dataset, nrows=no_of_rows)
    return dataFrame

#Function that takes in a dataFrame type object and process it to include only the selected feature columns, and drop the rest
#subsequently convert all the data as integer format.
def process_data(dataFrame):
    feature_selection = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt','srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
    processed_data = pd.DataFrame(columns=feature_selection)
    processed_data = pd.concat([processed_data, dataFrame[dataFrame['is_booking'] == 1][feature_selection]])
    for column in processed_data:
        processed_data[column] = processed_data[column].astype(str).astype(int);
    X = processed_data
    Y = processed_data['hotel_cluster'].values
    print(processed_data.head());
    return X, Y;
#Function to create and plot correlation matrix to perform data analysis
def plot_and_print_corr(dataset):
    corr_mat = dataset.corr()
    fig, ax = plt.subplots()
    col_names = ['site_name','posa_continent','user_location_country','user_location_region','user_location_city','orig_destination_distance','user_id','is_mobile','is_package','channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market','is_booking','cnt','hotel_cluster']
    im, cbar = heatmap(corr_mat, col_names, col_names, ax=ax,cmap="YlGn", cbarlabel="Correlation")
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    fig.tight_layout()
    plt.show()

dataFrame = read_csv(37670293,False)
plot_and_print_corr(dataFrame)

    
