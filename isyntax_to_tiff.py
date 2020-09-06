# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2020 Koninklijke Philips N.V. All Rights Reserved.

# A copyright license is hereby granted for redistribution and use of the
# Software in source and binary forms, with or without modification, provided
# that the following conditions are met:
# • Redistributions of source code must retain the above copyright notice, this
#   copyright license and the following disclaimer.
# • Redistributions in binary form must reproduce the above copyright notice,
#   this copyright license and the following disclaimer in the documentation
#   and/ or other materials provided with the distribution.
# • Neither the name of Koninklijke Philips N.V. nor the names of its
#   subsidiaries may be used to endorse or promote products derived from the
#   Software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This sample code generates regular/big tiff for a desired ROI.
The common inputs to the file are:
  1. Path of the iSyntax file
  2. Regular/Big Tiff (if 0 is passed regular tiff is generated,
      whereas if 1 is passed big tiff is generated)
  3. NotSparse/Sparse (if 0 is passed sparse tiff is not generated,
      whereas if 1 is passed sparse tiff is generated)
  4. Start_level (The tiff file is generated from starting level.)
Eg:
Command for regular tiff from level 0:
  python isyntax_to_tiff.py <Isyntax file path> 0 0 0
Command for big tiff from level 0:
  python isyntax_to_tiff.py <Isyntax file path> 1 0 0
Command for regular sparse tiff from level 0:
  python isyntax_to_tiff.py <Isyntax file path> 0 1 0
Command for big sparse tiff from level 0:
  python isyntax_to_tiff.py <Isyntax file path> 1 1 0
IMPORTANT NOTE : This sample code uses libtiff.
 One needs to download it for the target platform.

On Windows:
 Download a suitable version of libtiff or build it from GitHub source.
 Required DLL's:
    libtiff-5.dll, libjpeg-62.dll, zlib.dll

On Ubuntu: To install libtiff, execute 'apt-get install -y libtiff5-dev'
On CentOS: To install libtiff, execute 'yum install -y libtiff'

Dependencies:
    Pip modules: numpy, libtiff
"""

import os
import argparse
import traceback
import numpy as np
from pixelengine import PixelEngine
from softwarerendercontext import SoftwareRenderContext
from softwarerenderbackend import SoftwareRenderBackend
from libtiff_interface import *


def write_tiff_tile(tiff_handle, offset, level, sparse, data, data_size, bb_list, region):
    """
    Save extracted regions as Patches to Disk
    :param tiff_handle: Tiff file handle
    :param offset: List of Offset Indices
    :param level: Level of Tile
    :param sparse: Sparse Flag
    :param data: Buffer
    :param data_size: Size of buffer
    :param bb_list: Bounding box list
    :param region : Current region
    :return: None
    """
    try:
        # check sparse and background tiles
        write_tile = (not sparse) or (region_within_data_envelope(bb_list, region.range))
        if not write_tile:
            return

        # Write Tile
        if LIBTIFF.TIFFWriteEncodedTile(tiff_handle, LIBTIFF.TIFFComputeTile
                                        (tiff_handle, offset[0], offset[1],
                                         level, 0), data, data_size) < 0:
            print("Error in generating TIFF")

    except RuntimeError:
        traceback.print_exc()


def region_within_data_envelope(bb_list, bb_range):
    """
    Method to check background tile
    :param bb_list: Data envelope list
    :param bb_range: Tile view range
    :return: outside_x or outside_y
    """
    x_min = bb_range[0]
    x_max = bb_range[1]
    y_min = bb_range[2]
    y_max = bb_range[3]

    for bounding_box in bb_list:
        bound_x_min = bounding_box[0]
        bound_x_max = bounding_box[1]
        bound_y_min = bounding_box[2]
        bound_y_max = bounding_box[3]

        outside = (x_max < bound_x_min) or (bound_x_max < x_min) \
            or (y_max < bound_y_min) or (bound_y_max < y_min)
        if not outside:
            return True
    return False


def tiff_tile_processor(pixel_engine, level, patches,
                        patch_identifier, tiff_file_handle, sparse):
    """
    Tiff Tile Processor
    :param pixel_engine: Object of pixel Engine
    :param tile_width: Tile Width
    :param tile_height: Tile Height
    :param level: Level
    :param patches: List of patches
    :param patch_identifier: Identifier list to map patches when fetched from pixel engine
    :param tiff_file_handle: Tiff file handle
    :param bb_list: Bounding Box List
    :param sparse: Sparse Flag
    :return: None
    """
    view = pixel_engine["in"]["WSI"].source_view
    # To query for raw pixel data (source view), the truncation level should be disabled.
    trunc_level = {0: [0, 0, 0]}
    view.truncation(False, False, trunc_level)

    bb_list = []
    if sparse == 1:
        bb_list = view.data_envelopes(level).as_rectangles()
    samples_per_pixel = 3  # As we queried RGB for Pixel Data
    patch_data_size = int((TIFF_TILE_WIDTH * TIFF_TILE_HEIGHT * samples_per_pixel))
    data_envelopes = view.data_envelopes(level)
    # Requesting all the patches together may result in large memory footprint.
    # One can choose to optimize it by
    # requesting the patches incrementally, one by one or in small batches.
    regions = view.request_regions(patches, data_envelopes, True, [255, 0, 0],
                                   pixel_engine.BufferType(0))
    remaining_regions = len(regions)
    while remaining_regions > 0:
        regions_ready = pixel_engine.wait_any()
        remaining_regions -= len(regions_ready)
        for region in regions_ready:
            # Find the index of obtained Region in Original PatchList
            patch_id = patch_identifier[regions.index(region)]
            x_spatial = patch_id[0]
            y_spatial = patch_id[1]
            patch = np.empty(int(patch_data_size)).astype(np.uint8)
            region.get(patch)
            # Set the spatial location to paste in the TIFF file
            x_value = x_spatial * TIFF_TILE_WIDTH
            y_value = y_spatial * TIFF_TILE_HEIGHT
            write_tiff_tile(tiff_file_handle, [x_value, y_value], level, sparse,
                            patch.ctypes.data, patch_data_size, bb_list, region)
            regions.remove(region)
            patch_identifier.remove(patch_id)


def calculate_tiff_dimensions(view, start_level):
    """
    Set the TIFF tile size
    Note that TIFF mandates tile size in multiples of 16
    Calculate the Image Dimension range from the View at the Start Level
    :param view: Source View
    :param start_level: Starting Level
    :return: tiff_dim_x, tiff_dim_y
    """
    x_start = view.dimension_ranges(start_level)[0][0]
    x_step = view.dimension_ranges(start_level)[0][1]
    x_end = view.dimension_ranges(start_level)[0][2]
    y_start = view.dimension_ranges(start_level)[1][0]
    y_step = view.dimension_ranges(start_level)[1][1]
    y_end = view.dimension_ranges(start_level)[1][2]
    range_x = x_end - x_start + x_step
    range_y = y_end - y_start + y_step

    # As the multi-resolution image pyramid in TIFF
    #  shall follow a down sample factor of 2
    # Normalize the Image Dimension from the coarsest level
    #  so that a downscale factor of 2 is maintained across levels
    # Size Normalization
    tiff_dim_x = round_up(range_x, TIFF_TILE_WIDTH * x_step)
    tiff_dim_y = round_up(range_y, TIFF_TILE_HEIGHT * y_step)
    return tiff_dim_x, tiff_dim_y


def round_up(value, multiple):
    """
    Round Up
    :param value: Value
    :param multiple: Mu,tiple
    :return: Rounded up value
    """
    result = value
    if (value % multiple) > 0:
        result = value + multiple - (value % multiple)
    return result


def create_tiff_from_isyntax(pixel_engine, tiff_file_handle, start_level,
                             num_levels, sparse):
    """
    Method to create tiff from isyntax file
    :param pixel_engine: Object of Pixel Engine
    :param tiff_file_handle: Tiff file handle
    :param start_level: Start level
    :param num_levels: max levels in isyntax file
    :param sparse: Sparse Flag
    :return: 0
    """
    view = pixel_engine["in"]["WSI"].source_view
    tiff_dim_x, tiff_dim_y = calculate_tiff_dimensions(view, start_level)

    #  Scanned Tissue Area
    # Level 0 represents 40x scan factor
    # So, in order save time and as per the requirement,
    #  one can start from a coarser resolution level say level 2 (10x)

    for level in range(start_level, num_levels, 1):
        print("Level: {}".format(str(level)))
        # Take starting point as the dimensionRange start on the View
        #  for a particular Level
        x_start = view.dimension_ranges(level)[0][0]
        y_start = view.dimension_ranges(level)[1][0]

        # As the index representation is always in Base Level i.e. Level0, but
        # the step size increase with level as (2**level)
        width_patch_level = TIFF_TILE_WIDTH * (2 ** level)
        height_patch_level = TIFF_TILE_HEIGHT * (2 ** level)

        num_patches_x = round_up(tiff_dim_x, width_patch_level) / width_patch_level
        num_patches_y = round_up(tiff_dim_y, height_patch_level) / height_patch_level

        print("      - Number of Tiles in X and Y directions {}, {}".format(str(num_patches_x),
                                                                            str(num_patches_y)))

        # Error Resilience: Just in case if the number of patches at a given level in either
        # direction is 0, no point in writing tiff directory
        if num_patches_x * num_patches_y <= 0:
            print("TIFF Directory Write bypassed")
            continue

        set_tiff_file_attributes(tiff_file_handle, [tiff_dim_x, tiff_dim_y], level, start_level)
        patches, patch_identifier = create_patch_list(int(num_patches_x), int(num_patches_y),
                                                      [x_start, y_start], level,
                                                      [width_patch_level, height_patch_level])
        # Extract and Write TIFF Tiles
        print("      - Requesting Patches.")
        tiff_tile_processor(pixel_engine, level,
                            patches, patch_identifier,
                            tiff_file_handle, sparse)
        print("      - Patches extracted to disk")
        tiff_file_handle.write_directory()
    return 0


def set_attribute(tiff_file_handle, key, value):
    """
    Set Tiff file attributes
    :param tiff_file_handle: Tiff file handle
    :param key: Associated key
    :param value: value of key
    :return: None
    """
    assert tiff_file_handle.set_field(key, value) == 1, \
        "could not set {} tag".format(str(key))


def set_tiff_file_attributes(tiff_file_handle, tiff_dim, level, start_level):
    """
    Setting tiff file common attributes
    :param tiff_file_handle: Tiff file handle
    :param tiff_dim: TIFF Dimensions at level
    :param level: Current level
    :param start_level: STarting level of TIFF file
    :return: None
    """
    level_scale_factor = 2 ** level
    # For subdirectories corresponding to the multi-resolution pyramid, set the following
    # Tag for all levels but the initial level
    if level > start_level:
        set_attribute(tiff_file_handle, TIFFTAG_SUBFILETYPE, FILETYPE_REDUCEDIMAGE)

    # Setting TIFF file attributes
    set_attribute(tiff_file_handle, TIFFTAG_IMAGEWIDTH, int(tiff_dim[0] / level_scale_factor))
    set_attribute(tiff_file_handle, TIFFTAG_IMAGELENGTH, int(tiff_dim[1] / level_scale_factor))
    set_attribute(tiff_file_handle, TIFFTAG_TILEWIDTH, TIFF_TILE_WIDTH)
    set_attribute(tiff_file_handle, TIFFTAG_TILELENGTH, TIFF_TILE_HEIGHT)
    set_attribute(tiff_file_handle, TIFFTAG_BITSPERSAMPLE, BITSPERSAMPLE)
    set_attribute(tiff_file_handle, TIFFTAG_SAMPLESPERPIXEL, SAMPLESPERPIXEL)
    set_attribute(tiff_file_handle, TIFFTAG_PLANARCONFIG, PLANARCONFIG)
    set_attribute(tiff_file_handle, TIFFTAG_COMPRESSION, COMPRESSION_JPEG)
    set_attribute(tiff_file_handle, TIFFTAG_JPEGQUALITY, JPEGQUALITY)
    set_attribute(tiff_file_handle, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT)

    use_rgb = False  # Flag to choose bewteen RGB and YCbCr color model
    if use_rgb:
        set_attribute(tiff_file_handle, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB)
    else:
        set_attribute(tiff_file_handle, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_YCBCR)
        set_attribute(tiff_file_handle, TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB)
        assert tiff_file_handle.set_field(TIFFTAG_YCBCRSUBSAMPLING, YCBCRHORIZONTAL,
                                          YCBCRVERTICAL) == 1, "could not set YCbCr subsample tag"


def create_patch_list(num_patches_x, num_patches_y, starting_indices, level, patch_size):
    """
    Method to create patches list and patch identifier list
    :param num_patches_x: Number of patches in x
    :param num_patches_y: Number of patches in y
    :param starting_indices: Starting indices
    :param level: Level
    :param patch_size: Size of patch
    :return: list of patches, patch_identifier
    """
    patches = []
    patch_identifier = []
    y_spatial = 0
    for y_counter in range(num_patches_y):
        y_patch_start = starting_indices[1] + (y_counter * patch_size[1])
        y_patch_end = y_patch_start + patch_size[1]
        x_spatial = 0
        for x_counter in range(num_patches_x):
            x_patch_start = starting_indices[0] + (x_counter * patch_size[0])
            x_patch_end = x_patch_start + patch_size[0]
            patch = [x_patch_start, x_patch_end - 2 ** level, y_patch_start,
                     y_patch_end - 2 ** level, level]
            patches.append(patch)
            patch_identifier.append([x_spatial, y_spatial])
            x_spatial += 1
        y_spatial += 1
    return patches, patch_identifier


def encode_file_path(file_path):
    """
    Method to encode file_path as per python version
    :param file_path: file_path
    :return: file_path
    """
    file_path = bytes(file_path, encoding='utf-8')
    return file_path


def get_tiff_handle(tiff_type, input_file, sparse):
    """
    Method to generate tiff file handle
    :param tiff_type: Type of tiff
    :param input_file: Input file
    :param sparse: Sparse flag
    :return: tiff_file_handle
    """
    file_name = ".tiff"
    if sparse:
        file_name = "_sparse{}".format(file_name)
    image_name = os.path.splitext(os.path.basename(input_file))[0]

    if tiff_type == 0:
        print("Regular Tiff")
        file_path = ".{}{}{}".format(os.path.sep, image_name,file_name)
        file_path = encode_file_path(file_path)
        tiff_file_handle = TIFF.open(file_path, mode=b'w')
    elif tiff_type == 1:
        print("Big Tiff")
        file_path = ".{}{}_BIG{}".format(os.path.sep, image_name, file_name)
        file_path = encode_file_path(file_path)
        tiff_file_handle = TIFF.open(file_path, mode=b'w8')

    return tiff_file_handle


def main():
    """
    Main Method
    :return: None
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="""
    This sample code generates regular/big tiff for a desired ROI.
    The common inputs to the file are:
      1. Path of the iSyntax file
      2. Regular/Big Tiff (if 0 is passed regular tiff is generated,
          whereas if 1 is passed big tiff is generated)
      3. Write Sparse Tiles/Do not write Sparse Tiles (if 0 is passed sparse tiff is not generated,
          whereas if 1 is passed sparse tiff is generated)
         Note: Sparse Tiles corresponds to the background tiles, which does not belong to any data envelopes.
      4. Start_level (The tiff file is generated from starting level.)
    Eg:
    Command for regular tiff from level 0:
      python isyntax_to_tiff.py <Isyntax file path> 0 0 0
    Command for big tiff from level 0:
      python isyntax_to_tiff.py <Isyntax file path> 1 0 0
    Command for regular sparse tiff from level 0:
      python isyntax_to_tiff.py <Isyntax file path> 0 1 0
    Command for big sparse tiff from level 0:
      python isyntax_to_tiff.py <Isyntax file path> 1 1 0
    IMPORTANT NOTE : This sample code uses libtiff.
     One needs to download it for the target platform.

    On Windows:
     Download a suitable version of libtiff or build it from GitHub source.
     Required DLL's:
        libtiff-5.dll, libjpeg-62.dll, zlib.dll

    On Ubuntu: To install libtiff, execute 'apt-get install -y libtiff5-dev'
    On CentOS: To install libtiff, execute 'yum install -y libtiff'

    Dependencies:
        Pip modules: numpy, libtiff
    """)
    parser.add_argument("input", help="Image File")
    parser.add_argument("tif", help="TIFF/BIGTIFF")
    parser.add_argument("sparse", help="Write Sparse Tiles = 0, Do Not Write Sparse Tiles = 1")
    parser.add_argument("startlevel", help="Starting Level")
    args = parser.parse_args()
    input_file = args.input

    # Initializing the pixel engine
    render_context = SoftwareRenderContext()
    render_backend = SoftwareRenderBackend()
    pixel_engine = PixelEngine(render_backend, render_context)
    pixel_engine["in"].open(input_file)
    start_level = args.startlevel
    start_level = int(start_level[0])
    tiff_type = args.tif
    tiff_type = int(tiff_type[0])
    sparse = args.sparse
    sparse = int(sparse[0])
    if not (0 <= sparse <= 1 and 0 <= tiff_type <= 1):
        print("Invalid arguments passed")
        return
    tiff_file_handle = get_tiff_handle(tiff_type, input_file, sparse)

    num_derived_levels = pixel_engine["in"]["WSI"].source_view.num_derived_levels
    if 0 <= start_level <= num_derived_levels:
        print("Generating TIFF, Please Wait.....")
        result = create_tiff_from_isyntax(pixel_engine, tiff_file_handle, start_level,
                                          int(num_derived_levels) + 1,
                                          sparse)
        # Close the TIFF file handle.
        LIBTIFF.TIFFClose(tiff_file_handle)
        if result == 0:
            print("TIFF Successfully Generated")
        else:
            print("Error in generating TIFF")
    else:
        print("Invalid start_level Input")
    pixel_engine["in"].close()


if __name__ == '__main__':
    main()
