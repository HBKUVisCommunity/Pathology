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
Tile Processor
"""
try:
    import numpy as np
    from PIL import Image, ImageTk
    from constants import Constants
    from pixel_engine_connector import PixelEngineConnector
    from tile import Tile
    import io
except ImportError as exp:
    print(exp)


class TileProcessor:
    """
    This class is responsible for processing tiles as per the request of tkinter display
    bounding box.
    """
    def __init__(self, input_file, args_level, args_backend, args_display_view):
        """
        The constructor initializes pixel engine connector class.
        :param input_file: IsyntaxFilePath
        :param args_backend: The requested backend
        """
        self._tile_list = []
        self._prev_tile_list = []
        self._level_info = []
        self._level = 0
        self.pixel_engine_connector = PixelEngineConnector(input_file, args_backend,
                                                           args_display_view)
        self.create_level_info(args_level)
        # self.generate_image_properties()

    def get_sub_image(self, sub_image, sub_image_size):
        """
        This method returns the sub image (LABEL or MACRO image) data.
        :param sub_image: Requested sub image LABEL or MACRO image
        :param sub_image_size: Requested sub image size
        :return: Requested sub image data
        """
        isyntax_facade_obj = self.pixel_engine_connector.get_isyntax_facade_obj()
        for image in range(isyntax_facade_obj.num_images):
            image_type = isyntax_facade_obj[image].image_type
            if image_type == sub_image:
                sub_image = Image.open(io.BytesIO(isyntax_facade_obj[image].image_data))
                sub_image = sub_image.resize((sub_image_size[0],
                                              sub_image_size[1]), Image.ANTIALIAS)
                sub_image = ImageTk.PhotoImage(sub_image)
        return sub_image

    def get_max_levels(self):
        """
        Queries maximum number of levels in isyntax file through pixel_engine_connector
        :return: Maximum number of levels of isyntax file
        """
        return self.pixel_engine_connector.get_max_levels()

    def get_level(self):
        """
        Method to return the current level
        :return: Returns the current level
        """
        return self._level

    def set_level(self, new_level):
        """
        Method to set level
        :param new_level: Desired level
        :return: None
        """
        self._level = new_level

    def set_prev_tile_list(self, new_tile_list):
        """
        Method to update prev_tile_list
        :param new_tile_list: Updated new tile list
        :return: None
        """
        self._prev_tile_list = new_tile_list

    def get_level_info_list(self):
        """
        Method to get level info list
        :return: Level info list which holds the tiles of all the levels.
        """
        return self._level_info

    def remove_tiles(self, redundant_tiles):
        """
        Removes the redundant tiles from canvas
        :param redundant_tiles: List of redundant tiles which needs to be removed
        :return:None
        """
        for row_col in redundant_tiles:
            if row_col in self._level_info[self.get_level()][3]:
                tile = self._level_info[self.get_level()][4][row_col]
                if tile.get_image() is not None:
                    tile.set_image(None)

    def processing_tiles(self, bounding_box_range, current_row_col,flg):
        """
        Processes bounding box and check for redundant tiles, common tiles, tiles to be fetched.
        :param bounding_box_start_row: Starting row of bounding box
        :param bounding_box_start_col: Starting column of bounding box
        :param bounding_box_end_row: Ending row of bounding box
        :param bounding_box_end_col: Ending column of bounding box
        :param current_row: Current tile, row in which event occurred
        :param current_col: Current tile, column in which event occurred
        :return: List of new tiles that need to be displayed
        """
        # Preparing a list of tiles that lies in the bounding box
        current_tiles = []
        for col in range(bounding_box_range[1], bounding_box_range[3] + 1):
            for row in range(bounding_box_range[0], bounding_box_range[2] + 1):
                current_tiles.append((row, col))

        # Checking for intersecting tiles from previous update
        previous_tiles = self._prev_tile_list

        # common between previous and current
        common_tiles = list(set(previous_tiles) & set(current_tiles))

        # the tiles which are not required in current
        redundant_tiles = list(set(previous_tiles) ^ set(common_tiles))
        # Removing unused tiles from cache
        self.remove_tiles(redundant_tiles)
        self._prev_tile_list = current_tiles

        # rearrange to get highlighted tile first
        tiles_tobe_fetched = self.rearrange_current_tiles(current_tiles, current_row_col)

        self._tile_list = []
        self.update_list(tiles_tobe_fetched,flg)

        display_list = self.tile_extract(self._tile_list, self._level)
        return display_list

    def update_list(self, tiles_tobe_fetched,flg=False):
        """
        Prepares tile list that needs to be fetched from pixel engine
        :param tiles_tobe_fetched: List of tiles that needs to be fetched from the pixel engine
        :return: None
        """
        tile_list = []
        for row_col in tiles_tobe_fetched:
            if row_col in self._level_info[self._level][3]:
                tile = self._level_info[self._level][4][row_col]
                if tile.get_image() is None or flg is True:
                    tile_list.append(tile)
        self._tile_list = tile_list

    def tile_extract(self, tile_list, level):
        """
        Method to extract new tiles
        :param tile_list: The list of tiles that needs to be extracted
        :param level: The level at which these tiles should be extracted
        :return: Data of tiles that are fetched from the system.
        """
        if not tile_list:
            return 0
        tiles = []
        display_list = []
        tile_size = Constants.tile_width * Constants.tile_height * Constants.sample_per_pixel

        for tile in tile_list:
            tiles.append(tile.get_view_range())

        regions = self.pixel_engine_connector.get_source_view().\
            request_regions(tiles,
                            self.pixel_engine_connector.get_source_view().data_envelopes(level),
                            False, [254, 254, 254])
        remaining_regions = len(regions)
        while remaining_regions > 0:
            regions_ready = self.pixel_engine_connector.get_pe().wait_any()
            remaining_regions -= len(regions_ready)
            for region in regions_ready:
                level = region.range[4]
                tile = self.get_tile(tile_list, region.range)
                new_tile = np.empty(tile_size, dtype=np.uint8)
                region.get(new_tile)
                regions.remove(region)
                tile.set_image(
                    (Image.frombuffer('RGB', (Constants.tile_width, Constants.tile_height),
                                      new_tile, 'raw', 'RGB', 0, 1)))
                tile.set_image(ImageTk.PhotoImage(tile.get_image()))
                paste_can_img_x = int((region.range[0] / 2 ** level) - self._level_info[level][1] /
                                      2 ** level)
                paste_can_img_y = int((region.range[2] / 2 ** level) - self._level_info[level][2] /
                                      2 ** level)
                display_list.append([paste_can_img_x, paste_can_img_y, tile.get_image()])
        return display_list

    @staticmethod
    def rearrange_current_tiles(tiles_tobe_fetched, current_row_col):
        """
        Setting current tile as first tile in the list
        :param tiles_tobe_fetched: List of tiles that needs to fetched
        :param current_row: Current tile, row in which event occurred
        :param current_col: Current tile, column in which event occurred
        :return: Rearranged tiles to be fetched
        """
        if (current_row_col[0], current_row_col[1]) in tiles_tobe_fetched:
            tiles_tobe_fetched.insert(0, tiles_tobe_fetched.pop(
                tiles_tobe_fetched.index((current_row_col[0], current_row_col[1]))))
        return tiles_tobe_fetched

    def create_level_info(self, args_level):
        """
        Method to create tile as object of LevelTileInfo
        This method creates view range for every tile and create tile as an object of LevelTileInfo.
        :return: None
        """
        level_info = []
        for level in range(self.pixel_engine_connector.get_max_levels()):
            tiles = []
            x_start = self.pixel_engine_connector.get_source_view().dimension_ranges(level)[0][0]
            x_end = (self.pixel_engine_connector.get_source_view().dimension_ranges(level)[0][2] +
                     Constants.tile_width * (2 ** level))
            y_start = self.pixel_engine_connector.get_source_view().dimension_ranges(level)[1][0]
            y_end = (self.pixel_engine_connector.get_source_view().dimension_ranges(level)[1][2] +
                     Constants.tile_height * (2 ** level))
            row_col = []
            col = 0
            for y_pix in range(y_start, y_end, Constants.tile_height * (2 ** level)):
                row = 0
                for x_pix in range(x_start, x_end, Constants.tile_width * (2 ** level)):
                    view_range = [x_pix,
                                  ((x_pix + Constants.tile_width * (2 ** level)) - 2 ** level),
                                  y_pix,
                                  ((y_pix + Constants.tile_height * (2 ** level)) - 2 ** level),
                                  level]
                    tiles.append(Tile(view_range, None))
                    row_col.append((row, col))
                    row = row + 1
                col = col + 1
                if 0 <= args_level <= self.pixel_engine_connector.get_max_levels():
                    self._level = args_level
                else:
                    self._level = level
                # self._level = level
            level_info.append((level, x_start, y_start, row_col,
                               dict(zip(row_col, tiles))))
        self._level_info = level_info

    @staticmethod
    def get_tile(tile_list, view_range):
        """
        This method returns tile object for the given view range.
        :param tile_list: The list of tiles that needs to be extracted.
        :param view_range: The view range of the tile
        :return: Tile object for the view range requested.
        """
        for tile in tile_list:
            if view_range == tile.get_view_range():
                return tile
        return -1
