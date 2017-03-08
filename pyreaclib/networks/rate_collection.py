# Common Imports
from __future__ import print_function

import glob
import os

import networkx as nx
import matplotlib.pyplot as plt

# Import Rate
from pyreaclib.rates import Rate

class RateCollection(object):
    """ a collection of rates that together define a network """

    def __init__(self, rate_files=None, rates=None):
                 
        """
        rate_files are the files that together define the network.
        This can be any iterable or single string, and can include
        wildcards.

        rates is an optional argument. If it is supplied it
        should be an iterable or single instance of Rate
        objects and the appropriate RateCollection will be
        constructed.
        
        If nothing is supplied for either rate_files or rates, 
        then the RateCollection will be empty but you can add
        Rate objects using the RateCollection.add() method.
        """
        self.pyreaclib_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.files = []
        self.rates = []

        if rate_files:
            if type(rate_files) is str:
                rate_files = [rate_files]
            self.get_rates_from_files(rate_files)
            self.organize_rates()

    def add(self, rate):
        """ Given a Rate object, add it to the collection. """
        if isinstance(rate, Rate):
            self.rates.append(Rate)
            self.organize_rates()
        else:
            print('The argument to RateCollection.add should be a Rate object')
        
    def organize_rates(self):
        self.get_unique_nuclei()
        self.map_rates_to_nuclei()
        self.sort_rates()

    def get_rates_from_files(self, rate_files):
        # get the rates in the list of rate files
        self.pyreaclib_rates_dir = os.path.join(self.pyreaclib_dir, 'rates')
        exit_program = False
        for p in rate_files:
            # check to see if the rate file is in the working dir
            fp = glob.glob(p)
            if fp:
                self.files += fp
            else:
                # check to see if the rate file is in pyreaclib/reaclib-rates
                fp = glob.glob(os.path.join(self.pyreaclib_rates_dir, p))
                if fp:
                    self.files += fp
                else: # Notify of all missing files before exiting
                    print('ERROR: File {} not found in {} or the working directory!'.format(
                        p,self.pyreaclib_rates_dir))
                    exit_program = True 
        if exit_program:
            exit()

        for rf in self.files:
            try:
                self.rates.append(Rate(rf))
            except:
                print("Error with file: {}".format(rf))
                raise

    def get_unique_nuclei(self):
        # get the unique nuclei
        u = []
        for r in self.rates:
            t = set(r.reactants + r.products)
            u = set(list(u) + list(t))
        self.unique_nuclei = sorted(u)

    def map_rates_to_nuclei(self):
        # NOTE: get_unique_nuclei should be called first.
        # now make a list of each rate that touches each nucleus
        # we'll store this in a dictionary keyed on the nucleus
        self.nuclei_consumed = {}
        self.nuclei_produced = {}

        for n in self.unique_nuclei:
            self.nuclei_consumed[n] = []
            for r in self.rates:
                if n in r.reactants:
                    self.nuclei_consumed[n].append(r)

            self.nuclei_produced[n] = []
            for r in self.rates:
                if n in r.products:
                    self.nuclei_produced[n].append(r)

    def sort_rates(self):
        # Re-order self.rates so Reaclib rates come first,
        # followed by Tabular rates. This is needed if
        # reaclib coefficients are targets of a pointer array
        # in the Fortran network.
        # It is desired to avoid wasting array size
        # storing meaningless Tabular coefficient pointers.
        self.rates = sorted(self.rates,
                            key = lambda r: r.chapter=='t')

        self.tabular_rates = []
        self.reaclib_rates = []
        for n,r in enumerate(self.rates):
            if r.chapter == 't':
                self.tabular_rates.append(n)
            elif type(r.chapter)==int:
                self.reaclib_rates.append(n)
            else:
                print('ERROR: Chapter type unknown for rate chapter {}'.format(
                    str(r.chapter)))
                exit()

    def print_network_overview(self):
        for n in self.unique_nuclei:
            print(n)
            print("  consumed by: ")
            for r in self.nuclei_consumed[n]:
                print("     {}".format(r.string))

            print("  produced by: ")
            for r in self.nuclei_produced[n]:
                print("     {}".format(r.string))

            print(" ")
                
    def write_network(self):
        print('To create network integration source code, use a class that implements a specific network type.')
        return
                
    def plot(self):
        G = nx.DiGraph()
        G.position={}
        G.labels = {}

        plt.plot([0,0], [8,8], 'b-')

        # nodes
        for n in self.unique_nuclei:
            G.add_node(n)
            G.position[n] = (n.N, n.Z)
            G.labels[n] = r"${}$".format(n.pretty)

        # edges
        for n in self.unique_nuclei:
            for r in self.nuclei_consumed[n]:
                for p in r.products:
                    G.add_edges_from([(n, p)])


        nx.draw_networkx_nodes(G, G.position,
                               node_color="0.5", alpha=0.4,
                               node_shape="s", node_size=1000, linewidth=2.0)
        nx.draw_networkx_edges(G, G.position, edge_color="0.5")
        nx.draw_networkx_labels(G, G.position, G.labels, 
                                font_size=14, font_color="r", zorder=100)

        plt.xlim(-0.5,)
        plt.xlabel(r"$N$", fontsize="large")
        plt.ylabel(r"$Z$", fontsize="large")

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.show()

    def __repr__(self):
        string = ""
        for r in self.rates:
            string += "{}\n".format(r.string)
        return string

class RateLibrary(RateCollection):
    """
    The RateLibrary class holds a snapshot of the
    entire Reaclib library and can be helpful for
    creating a custom RateCollection instance
    and searching for reactions.

    The library_file argument can be passed pointing
    to a Reaclib snapshot file. The snapshot may be
    for any Reaclib version, but must be in Reaclib2 format.
    """
    def __init__(self, library_file = '20170303ReaclibV2.2'):
        super(RateLibrary, self).__init__()
        self.library_file = library_file
        self.parse_library_file()
        self.organize_rates()
    
    def parse_library_file(self):
        """ 
        Parse a standalone Reaclib library
        File should be in Reaclib v2 format though
        the data can be any Reaclib version.
        """
        # read in the file, parse the different Rates
        f = open(self.library_file, "r")
        lines = f.readlines()
        f.close()
        
        ratelist = self.get_library_rates(lines)
        for r in ratelist:
            self.add(r)

    def get_library_rates(self, lines):
        """ 
        Given a list of lines (lines),
        extract the rates as Rate objects
        and return them in a list.
        """
        ratelist = []
        # Try to read rates for as long as possible
        while(True):
            ratedata, lines = Rate.rate_from_lines(lines)
            if ratedata:
                ratelist.append(Rate(ratedata=ratedata))
            else:
                break
        return ratelist
        
                
        
