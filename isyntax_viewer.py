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
iSyntax Viewer - view iSyntax file
To execute this program
pass the path of the iSyntax image file as command line argument.
eg:
    Source View:
        python isyntax_viewer.py "<PATH_OF_ISYNTAX_FILE>"
    Display View:
        python isyntax_viewer.py "<PATH_OF_ISYNTAX_FILE>" --display_view
Dependencies:
    numpy, tkinter, pillow, six
"""


try:
    import argparse
    from tkinter_display import TkinterDisplay
    from pixel_engine_connector import BACKENDS
except ImportError as exp:
    print(exp)


def main():
    """
    Path of iSyntax file is taken as input in command line argument.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="""
iSyntax Viewer - view iSyntax file
To execute this program
pass the path of the iSyntax image file as command line argument.
eg:
    Source View:
        python isyntax_viewer.py "<PATH_OF_ISYNTAX_FILE>"
    Display View:
        python isyntax_viewer.py "<PATH_OF_ISYNTAX_FILE>" --display_view """)
    parser.add_argument("IsyntaxFilePath",
                        help="Enter path of iSyntax file enclosed in double quotes")
    parser.add_argument("-b", "--backend",
                        choices=[backend.name for backend in BACKENDS], nargs='?',
                        default='SOFTWARE', help="select renderbackend")
    parser.add_argument("-l", "--level", default=-1, help="Enter isyntax display level")
    parser.add_argument("--display_view", action='store_true', help="for display view")

    args = parser.parse_args()
    print("Initializing viewer, please wait...")

    TkinterDisplay(args.IsyntaxFilePath, int(args.level), args.backend, args.display_view)


if __name__ == '__main__':
    main()
