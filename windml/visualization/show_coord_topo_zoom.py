"""
Copyright (c) 2013,
Fabian Gieseke, Justin P. Heinermann, Oliver Kramer, Jendrik Poloczek,
Nils A. Treiber
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    Neither the name of the Computational Intelligence Group of the University
    of Oldenburg nor the names of its contributors may be used to endorse or
    promote products derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import math
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.basemap import Basemap, shiftgrid, cm
    from mpl_toolkits.axes_grid.inset_locator import mark_inset
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
except ImportError:
    try:
        import importlib
        mpl_toolkits = importlib.import_module('mpl_toolkits')
        from mpl_toolkits.basemap import Basemap, shiftgrid, cm
        from mpl_toolkits.axes_grid.inset_locator import mark_inset
        from mpl_toolkits.axes_grid.inset_locator import inset_axes
    except ImportError:
        raise Exception('Could not load mpl_toolkits')


def show_coord_topo_zoom(windpark, show = True):
    """Plot the topology of a given windpark

    Topographic Map with farms
    see: http://matplotlib.org/basemap/users/examples.html
    Basemap

    Parameters
    ----------

    windpark : Windpark
               A given windpark to show the topology.
    """

    plt.clf()

    turbines = windpark.get_turbines()
    target = windpark.get_target()
    radius = windpark.get_radius()

    #pack latitude and longitude in lists
    rel_input_lat = []
    rel_input_lon = []
    for row in turbines:
        rel_input_lat.append(np.float64(row.latitude))
        rel_input_lon.append(np.float64(row.longitude))

    targetcoord = [0.0, 0.0]
    targetcoord[0] = np.float64(target.latitude)
    targetcoord[1] = np.float64(target.longitude)

    graddiff = (800/111.0) + 0.5  # degree in km... 800km

    fig = plt.figure(figsize=(11.7,8.3))
    plt.title("Location of the farm")

    #Custom adjust of the subplots
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
    ax = plt.subplot(111)

    m = Basemap(projection='lcc', lon_0=targetcoord[1], lat_0=targetcoord[0],\
        llcrnrlon = targetcoord[1]-graddiff*2, llcrnrlat = targetcoord[0]-graddiff ,\
        urcrnrlon = targetcoord[1]+graddiff, urcrnrlat = targetcoord[0]+graddiff ,\
        rsphere=6371200., resolution = None, area_thresh=1000)

    # Target
    x_target,y_target = m(targetcoord[1],targetcoord[0])
    # Input Farms
    rel_inputs_lon, rel_inputs_lat = m(rel_input_lon, rel_input_lat)

    # labels = [left,right,top,bottom]
    parallels = np.arange(int(targetcoord[0]-7), int(targetcoord[0]+7), 2.)
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(int(targetcoord[1]-7), int(targetcoord[1]+7), 2.)
    m.drawmeridians(meridians,labels=[True,False,False,True])


    # plot farms in the radius
    m.plot(x_target, y_target, 'bo')
    # m.plot(rel_inputs_lon, rel_inputs_lat, 'r*')

    #m.bluemarble()
    m.shadedrelief()

    # we define the inset_axes, with a zoom of 2 and at location 2 (upper left corner)
    # axins = zoomed_inset_axes(ax, 40, loc=2)
    axins = inset_axes(ax,
                        width="65%", # width = 30% of parent_bbox
                        height="65%", # height : 1 inch
                        loc=3)

    # plot farms in the radius
    m.plot(x_target, y_target, 'bo')
    m.plot(rel_inputs_lon, rel_inputs_lat, 'r*')

    #m.bluemarble()
    m.shadedrelief()
    #m.etopo()
    #m.drawcoastlines()

    graddiff_park = (radius/(111.0*0.7))  # degree in km
    x,y = m(targetcoord[1]-graddiff_park,targetcoord[0]-graddiff_park)
    x2,y2 = m(targetcoord[1]+graddiff_park,targetcoord[0]+graddiff_park)

    axins.set_xlim(x,x2) # and we apply the limits of the zoom plot to the inset axes
    axins.set_ylim(y,y2) # idem
    plt.xticks(visible=False) # we hide the ticks
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.9", linewidth = 3) # we draw the "zoom effect" on the main map (ax), joining cornder 1 & 3

    plt.title("Selected Wind Farms")
    if(show):
        plt.show()

