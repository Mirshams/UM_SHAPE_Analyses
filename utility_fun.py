# Useful functions for getting colormap data
# 7.29.2021

import matplotlib.pyplot as plt
import numpy as np
import pydicom

def Dim_Lookup(pixel_array):
    """
    This function uses the DICOM file header to look up the width and height of each frame and returns them.
    :param
        pixel_array: Still frame: (n_x, n_y, 3), or Cine Loop: (n_frame, n_x, n_y, 3)
    :return:
        frame_dim = (n_x, n_y)
    """
    full_dim = pixel_array.shape
    if len(full_dim) == 3:
        frame_dim = tuple(full_dim[0:2])
        return frame_dim
    elif len(full_dim) == 4:
        frame_dim = tuple(full_dim[1:3])
        return frame_dim
    else:
        print("Pixel_Array has the wrong dimensions!" + str(full_dim))

def Mask_Color(pixel_array):
    """
            This function masks any point in the frame where R = G = B to R = G = B = 0
            :param
                frame: a Single Frame of ultrasound in the RGB format
            :return:
                mask: logical array the same shape as frame, where all grey points are 0
            """
    assert len(pixel_array.shape) in [3, 4]
    mask = np.zeros(pixel_array.shape)
    Comp1 = np.not_equal(pixel_array[..., 0], pixel_array[..., 1])
    Comp2 = np.not_equal(pixel_array[..., 0], pixel_array[..., 2])
    Comp3 = np.not_equal(pixel_array[..., 1], pixel_array[..., 2])
    mask[..., 0] = np.logical_or(np.logical_or(Comp1, Comp2), Comp3)
    mask[..., 1] = mask[..., 0]
    mask[..., 2] = mask[..., 0]
    return mask


def Masked_BMode(pixel_array):
    mask = Mask_Color(pixel_array)
    masked_array = np.multiply(mask, pixel_array).astype(pixel_array.dtype)
    return masked_array


def Find_ColorBar(pixel_array, est_loc=(0, 600, 200, 800)):
    """
    This function isolates the color bar which could be used for the color map
    :param
    pixel_array: the full pixel_array from the DICOM cine loop. If there is more than one frame, the first frame will
    be used for extracting the color bar
    est_loc: an array in form of [row_0, col_0, row_f, col_f]
    :return
    masked_bar: just the color bar found in the specified location in est_loc
    """
    assert len(pixel_array.shape) in [3, 4]
    if len(pixel_array.shape) == 4:
        frame = pixel_array[1, ...]
    else:
        frame = pixel_array
    mask = Mask_Color(frame)
    row0 = est_loc[0]
    row1 = est_loc[2]
    col0 = est_loc[1]
    col1 = est_loc[3]
    small_mask = mask[row0:row1, col0:col1, 1]
    small_array = frame[row0:row1, col0:col1]
    # ------------ find where the colorbar resides in the specified region and trim the specified region ------------
    boz = np.where(np.sum(small_mask, axis=0, keepdims=True) != 0)
    col00 = boz[1][0]
    col11 = boz[1][-1]
    # ^ Add all rows together and find the first and last column where the color bar exists
    boz = np.where(np.sum(small_mask, axis=1, keepdims=True) != 0)
    row00 = boz[0][0]
    row11 = boz[0][-1]
    # ^ Add all columns together and find the first and last row where the color bar exists
    small_mask = mask[row0 + row00:row0 + row11, col0 + col00:col0 + col11, ]
    small_array = small_array[row00:row11, col00:col11]
    masked_bar = np.multiply(small_mask, small_array).astype(pixel_array.dtype)
    if len(masked_bar) < 128:
        pad_zeros = np.zeros((128-len(masked_bar), masked_bar.shape[1], masked_bar.shape[2]))
        masked_bar = np.concatenate((masked_bar, pad_zeros), axis=0)
    elif (len(masked_bar) > 128) & (len(masked_bar) < 256):
        pad_zeros = np.zeros((128 - len(masked_bar), masked_bar.shape[1], masked_bar.shape[2]))
        masked_bar = np.concatenate((masked_bar, pad_zeros), axis=0)
    masked_bar = masked_bar.astype(pixel_array.dtype)

    # ------------ show the extracted color bar --------------------------------------------------------------------
    fig_4 = plt.figure(figsize=(5, 4), dpi=100)
    axes_4 = fig_4.add_axes([0.1, 0.1, 0.4, 0.9])
    axes_4.imshow(masked_bar, cmap='gray')
    axes_4.axis('off')

    return masked_bar

def amplitude_find(Pixel_Values, Color_Pallet, DR = 60, Method = 'meanRGB'):
    """
    This function returns the amplitude values in dB based on the input parameters and the pixel values in RGB

    :parameters
        Pixel_Values: Array of shape (n_pix, 3), or (n_rows, n_cols, 3), or (n_frames, n_rows, n_cols, 3) where
        3 is for RGB. Based on the RGB value for each pixel we want to return the amplitudes based on the Color_Pallet

        Color_Pallet: Array of shape (128 or 256, 3) which represent the color pallet for the RGB values.
        We are assuming the index of values in the color pallet's first index (0 - 127 or 0 - 255) are one-to-one linear
        with 0 to 60 dB.

        DR: The Dynamic Range

        Method: 'meanRGB' or 'useG'. meanRGB uses the mean value of the RGB values in both the color pallet and the
        Pixel_Values. useG only uses the Green value.
    :return:
        dB_out: the
    """
    # Check that the Pixel_Value array is in RGB format
    in_shape = Pixel_Values.shape
    if in_shape[-1] != 3:
        print(f"The input Pixel_Values is not in RGB format. Shape is {in_shape}!")
        return None

    # Check that Color_Pallet is in RGB format and is (n_colors, 3) shape
    cmap_shape = Color_Pallet.shape
    if (len(cmap_shape) != 2) | (cmap_shape[1] != 3):
        print(f"Color_Pallet is not in the correct format. Shape is {cmap_shape}!")
        return None

    # Flatten the Pixel values to an array of shape (n_pixs, 3)
    Pixels_Flat = Pixel_Values.reshape((-1, 3))

    if Method == "meanRGB":
        color_ref = np.mean(Color_Pallet, axis=1, keepdims=True)

        # Make sure color_ref is an increasing array
        if color_ref[0] > color_ref[-1]:
            color_ref = color_ref[::-1]

        Pixels_Mean = np.mean(Pixels_Flat, axis=1, keepdims=True)
        ind_vector = np.linspace(start=0, stop=1, endpoint=True, num=len(color_ref)).reshape(-1, 1)
        dB_out = np.interp(Pixels_Mean[:, 0], color_ref[:, 0], ind_vector[:, 0]).reshape(in_shape[0:-1]) * DR

    elif Method == "useG":
        color_ref = Color_Pallet[:, 1].reshape(-1, 1)
        # Make sure color_ref is an increasing array
        if color_ref[0] > color_ref[-1]:
            color_ref = color_ref[::-1]

        Pixels_Value = Pixels_Flat[:, 1].reshape(-1, 1)
        ind_vector = np.linspace(start=0, stop=1, endpoint=True, num=len(color_ref)).reshape(-1, 1)
        dB_out = np.interp(Pixels_Value[:, 0], color_ref[:, 0], ind_vector[:, 0]).reshape(in_shape[0:-1]) * DR

    else:
        dB_out = None

    amp_out = 10 ** (dB_out / 20)

    return (dB_out, amp_out, DR)

def create_color_pallet(red_array, green_array, blue_array, show_fig = False):
    """
    Take in the private fields for the color map from the ZS3 DICOM file and return the color pallet numpy array

    :param
        red_array: ds[0x0063, 1036].value
        green_array: ds[0x0063, 1037].value
        blue_array: ds[0x0063, 1038].value
        show_fig: if True, a Figure with the values of the color_pallet and the color map that they would make
    :return:
        color_pallet: Numpy array of shape (128 or 256, 3) where each row is the RGB value for a given index on the colormap.
    """
    clrmap_r = red_array.split('\\')
    clrmap_r = np.array(clrmap_r)
    clrmap_r = clrmap_r.astype(int).reshape(-1, 1)

    clrmap_g = green_array.split('\\')
    clrmap_g = np.array(clrmap_g)
    clrmap_g = clrmap_g.astype(int).reshape(-1, 1)

    clrmap_b = blue_array.split('\\')
    clrmap_b = np.array(clrmap_b)
    clrmap_b = clrmap_b.astype(int).reshape(-1, 1)

    if show_fig:
        # creating a colormap with arbitrary width
        clrmap_rr = np.tile(clrmap_r.reshape(1, -1), (20, 1))
        clrmap_gg = np.tile(clrmap_g.reshape(1, -1), (20, 1))
        clrmap_bb = np.tile(clrmap_b.reshape(1, -1), (20, 1))
        clrmap_img = np.stack((clrmap_rr, clrmap_gg, clrmap_bb), axis=2)

        fig = plt.figure(figsize=(8, 8))

        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax1.set_title("(0063, 1036:1038) DICOM Tags")
        ax1.plot(np.arange(len(clrmap_r)), clrmap_r, color='r')
        ax1.plot(np.arange(len(clrmap_g)), clrmap_g, color='g')
        ax1.plot(np.arange(len(clrmap_b)), clrmap_b, color='b')
        ax1.grid(True)
        ax1.set_ylabel('Value')
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        # colormap image
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        ax2.imshow(clrmap_img, origin='lower')
        ax2.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom edge are off
        ax2.set_xlabel('Index')

    color_pallet = np.concatenate((clrmap_r, clrmap_g, clrmap_b), axis=1)

    return color_pallet

def extract_region_data(pixel_array, regions, image_type='CEUS'):
    """
    Use (0018, 6011) Sequence of Ultrasound Regions to extract Contrast or B-Mode Images from the pixel_array
    :param
        pixel_array:
        regions:
        image_type: BMode (also B_Mode, bmode, or b_mode) or Contrast (also CEUS or contrast)
    :return:
        CEUS or B_Mode part of the pixel_array
    """
    if (image_type == 'BMode') | (image_type == 'B_Mode') | (image_type == 'bmode') | (image_type == 'b_mode'):
        region = regions[0]
    elif (image_type == 'contrast') | (image_type == 'CEUS') | (image_type == 'Contrast'):
        region = regions[1]

    image_width = region.RegionLocationMaxX1 - region.RegionLocationMinX0
    image_height = region.RegionLocationMaxY1 - region.RegionLocationMinY0

    if len(pixel_array.shape) == 3:
        new_pixel_array = pixel_array[region.RegionLocationMinY0:region.RegionLocationMaxY1,
                          region.RegionLocationMinX0:region.RegionLocationMaxX1, :]
    elif len(pixel_array.shape) == 4:
        new_pixel_array = pixel_array[:, region.RegionLocationMinY0:region.RegionLocationMaxY1,
                          region.RegionLocationMinX0:region.RegionLocationMaxX1, :]
    else:
        print(f"Wrong pixel_array input. Shape is {pixel_array.shape}")
        new_pixel_array = None

    return (new_pixel_array, image_width, image_height)

def cart_grid_and_roi_maker(region, roi_a, roi_b, roi_cntr=[np.nan, np.nan]):
    """
    Make Cartesian grid for an image given in region, use the grid to make roi
    Shape of ROI is ellipse
    Origin of Cartesian coordinate system is set as upper left corner of image

    Parameters:
    -----------
    region: pydicom.dataset
        Sequence of ultrasound region

    roi_cntr: Tuple of floats
        Center of the ROI w/o considering the offset, in m

    roi_a, roi_b: Float
        Semi major and minor axes of the elipse used as ROI

    nx_b, ny_b, nx_cntrst, ny_cntrst: Integers
        Number of grid poins (pixels) in x and y direction for B and
        Contrast modes

    Returns:
    --------
    x_b, y_b, x_cntrst, y_cntrst: 1D numpy array, float
        Vectors used in creating meshgrid

    X_b, Y_b, X_cntrst, Y_cntrst: 2D numpy array
        Mesh grid of cartesian grid

    roi_contours: 2D numpy array
        Contours of ellipses used as ROI

    """

    # Making x, y grid for scan-converted image
    image_width_pixel = (region.RegionLocationMaxX1
                         - region.RegionLocationMinX0)
    image_height_pixel = (region.RegionLocationMaxY1
                          - region.RegionLocationMinY0)
    image_width_cm = image_width_pixel * region.PhysicalDeltaX
    image_height_cm = image_height_pixel * region.PhysicalDeltaY

    x = np.linspace(0, image_width_cm, image_width_pixel, endpoint=False)
    y = np.linspace(0, image_height_cm, image_height_pixel, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    if np.isnan(roi_cntr[0]):
        roi_cntr = (image_width_cm / 2, roi_cntr[1])
    if np.isnan(roi_cntr[1]):
        roi_cntr = (roi_cntr[0], image_height_cm / 2)

    # ellipses defined based on above parameters
    roi_contours = (((xx - roi_cntr[0])/roi_a)**2 +
                    ((yy - roi_cntr[1])/roi_b)**2)

    return (x, y, xx, yy, roi_contours)
