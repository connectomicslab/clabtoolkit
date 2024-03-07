import numpy as np
import pandas as pd
import clabtoolkit.misctools as cltmisc


def _parc_tsv_table(codes, names, colors, tsv_filename):
    """
    Function to create a tsv table for parcellation

    Parameters
    ----------
    codes : list
        List of codes for the parcellation
    names : list
        List of names for the parcellation
    colors : list
        List of colors for the parcellation
    tsv_filename : str
        Name of the tsv file

    Returns
    -------
    tsv_df: pandas DataFrame
        DataFrame with the tsv table
    
    """
    
    # Table for parcellation
    # 1. Converting colors to hexidecimal string
    seg_hexcol = []
    nrows, ncols = colors.shape
    for i in np.arange(0, nrows):
        seg_hexcol.append(cltmisc.rgb2hex(colors[i, 0], colors[i, 1], colors[i, 2]))

    tsv_df = pd.DataFrame(
        {
            'index': np.asarray(codes),
            'name': names,
            'color': seg_hexcol
        }
    )
    #     print(bids_df)
    # Save the tsv table
    with open(tsv_filename, 'w+') as tsv_file:
        tsv_file.write(tsv_df.to_csv(sep='\t', index=False))
    
    return tsv_df


def tissue_seg_table(tsv_filename):
    """
    Function to create a tsv table for tissue segmentation

    Parameters
    ----------
    tsv_filename : str
        Name of the tsv file

    Returns
    -------
    seg_df: pandas DataFrame
        DataFrame with the tsv table
    
    """

    # Table for tissue segmentation
    # 1. Default values for tissues segmentation table
    seg_rgbcol = np.array([[172, 0, 0], [0, 153, 76], [0, 102, 204]])
    seg_codes = np.array([1, 2, 3])
    seg_names = ['cerebro_spinal_fluid', 'gray_matter', 'white_matter']
    seg_acron = ['CSF', 'GM', 'WM']

    # 2. Converting colors to hexidecimal string
    seg_hexcol = []
    nrows, ncols = seg_rgbcol.shape
    for i in np.arange(0, nrows):
        seg_hexcol.append(cltmisc.rgb2hex(seg_rgbcol[i, 0], seg_rgbcol[i, 1], seg_rgbcol[i, 2]))

    seg_df = pd.DataFrame(
        {
            'index': seg_codes,
            'name': seg_names,
            'abbreviation': seg_acron,
            'color': seg_hexcol
        }
    )
    # Save the tsv table
    with open(tsv_filename, 'w+') as tsv_file:
        tsv_file.write(seg_df.to_csv(sep='\t', index=False))
    
    return seg_df