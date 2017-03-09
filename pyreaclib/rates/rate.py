import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

from periodictable import elements

class Tfactors(object):
    """ precompute temperature factors for speed """

    def __init__(self, T):
        """ return the Tfactors object.  Here, T is temperature in Kelvin """
        self.T9 = T/1.e9
        self.T9i = 1.0/self.T9
        self.T913i = self.T9i**(1./3.)
        self.T913 = self.T9**(1./3.)
        self.T953 = self.T9**(5./3.)
        self.lnT9 = np.log(self.T9)

class SingleSet(object):
    """ a set in Reaclib is one piece of a rate, in the form

        lambda = exp[ a_0 + sum_{i=1}^5  a_i T_9**(2i-5)/3  + a_6 log T_9]

        A single rate in Reaclib can be composed of multiple sets
    """

    def __init__(self, a, label=None):
        """here a is iterable (e.g., list or numpy array), storing the
           coefficients, a0, ..., a6

        """
        self.a = a
        self.label = label


    def f(self):
        """
        return a function for this set -- note: Tf here is a Tfactors
        object
        """
        return lambda tf: np.exp(self.a[0] +
                                 self.a[1]*tf.T9i +
                                 self.a[2]*tf.T913i +
                                 self.a[3]*tf.T913 +
                                 self.a[4]*tf.T9 +
                                 self.a[5]*tf.T953 +
                                 self.a[6]*tf.lnT9)


    def set_string(self, prefix="set", plus_equal=False):
        """
        return a string containing the python code for this set
        """
        if plus_equal:
            string =  "{} += np.exp( ".format(prefix)
        else:
            string =  "{} = np.exp( ".format(prefix)
        string += " {}".format(self.a[0])
        if not self.a[1] == 0.0: string += " + {}*tf.T9i".format(self.a[1])
        if not self.a[2] == 0.0: string += " + {}*tf.T913i".format(self.a[2])
        if not self.a[3] == 0.0: string += " + {}*tf.T913".format(self.a[3])
        if not (self.a[4] == 0.0 and self.a[5] == 0.0 and self.a[6] == 0.0):
            string += "\n{}         ".format(len(prefix)*" ")
        if not self.a[4] == 0.0: string += " + {}*tf.T9".format(self.a[4])
        if not self.a[5] == 0.0: string += " + {}*tf.T953".format(self.a[5])
        if not self.a[6] == 0.0: string += " + {}*tf.lnT9".format(self.a[6])
        string += ")"
        return string

class Nucleus(object):
    """
    a nucleus that participates in a reaction -- we store it in a
    class to hold its properties, define a sorting, and give it a
    pretty printing string

    """
    def __init__(self, name):
        """ name should be as string as, e.g. he4 or li7 """
        self.raw = name

        # element symbol and atomic weight
        if name == "p":
            self.el = "H"
            self.A = 1
            self.short_spec_name = "h1"
        elif name == "d":
            self.el = "H"
            self.A = 2
            self.short_spec_name = "h2"
        elif name == "t":
            self.el = "H"
            self.A = 3
            self.short_spec_name = "h3"
        elif name == "n":
            self.el = "n"
            self.A = 1
            self.short_spec_name = "n"
        else:
            e = re.match("([a-zA-Z]*)(\d*)", name)
            self.el = e.group(1).title()  # chemical symbol

            self.A = int(e.group(2))
            self.short_spec_name = name

        # atomic number comes from periodtable
        i = elements.isotope("{}-{}".format(self.A, self.el))
        self.Z = i.number
        self.N = self.A - self.Z

        # long name
        if i.name == 'neutron':
            self.spec_name = i.name
        else:
            self.spec_name = '{}-{}'.format(i.name, self.A)

        # latex formatted style
        self.pretty = r"{{}}^{{{}}}\mathrm{{{}}}".format(self.A, self.el)

    def __repr__(self):
        return self.raw

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.raw == other.raw

    def __lt__(self, other):
        if not self.Z == other.Z:
            return self.Z < other.Z
        else:
            return self.A < other.A

class TabularRateData(object):
    """
    Container class for holding the data needed to 
    specify a tabular rate.
    """
    def __init__(self,
                 original_source="",
                 chapter=None,
                 reactants=[],
                 products=[],
                 table_file=None,
                 table_header_lines=None,
                 table_rhoy_lines=None,
                 table_temp_lines=None,
                 table_num_vars=None,
                 table_index_name=None):
        self.original_source = original_source
        self.chapter = chapter
        self.reactants = reactants
        self.products = products
        self.table_file = table_file
        self.table_header_lines = table_header_lines
        self.table_rhoy_lines = table_rhoy_lines
        self.table_temp_lines = table_temp_lines
        self.table_num_vars = table_num_vars
        self.table_index_name = table_index_name

class ReaclibRateData(object):
    """
    Container class for holding the data needed to
    specify a Reaclib rate.
    """
    def __init__(self,
                 original_source="",
                 chapter=None,
                 reactants=[],
                 products=[],
                 Q=None,
                 sets=[]):
        self.original_source = original_source
        self.chapter = chapter
        self.reactants = reactants
        self.products = products
        self.Q = Q
        self.sets = sets
        
class Rate(object):
    """ 
    A single Reaclib rate, which can be composed of multiple sets.

    If 'file' is supplied, look for a rate file by that name.

    If 'ratelines' is supplied, that should be a list of strings
    which are lines in a rate file (Reaclib v2-formatted)
    from the Reaclib rate library.

    If 'ratedata' is supplied, it should be an instance of either
    TabularRateData or ReaclibRateData and the appropriate Rate
    object will be constructed.
    """
    def __init__(self, file=None, ratelines=None, ratedata=None):
        if file:
            self.file = os.path.basename(file)
        self.chapter = None    # the Reaclib chapter for this reaction
        self.original_source = None   # the contents of the original rate file
        self.reactants = []
        self.products = []
        self.sets = []
        self.Q = 0.0
        self.string = ""
        self.pretty_string = r"$"
        self.prefactor = 1.0  # this is 1/2 for rates like a + a (double counting)
        self.inv_prefactor = 1

        # Tells if this rate is eligible for screening
        # using screenz.f90 provided by BoxLib Microphysics.
        # If not eligible for screening, set to None
        # If eligible for screening, then
        # Rate.ion_screen is a 2-element list of Nucleus objects for screening
        self.ion_screen = None 

        if not ratedata:
            if self.file:
                # Call rate file parser if its a standalone file.
                ratedata = Rate.parse_rate_file(self.file)
                idx = self.file.rfind("-")
                self.fname = self.file[:idx].replace("--","-").replace("-","_")
            elif lines:
                # Parse list of lines defining the rate
                ratedata,_ = Rate.rate_from_lines(ratelines)

        # Set Rate information
        if ratedata:
            if isinstance(ratedata, ReaclibRateData):
                self.original_source = ratedata.original_source
                self.chapter = ratedata.chapter
                self.reactants = ratedata.reactants
                self.products = ratedata.products
                self.Q = ratedata.Q
                self.sets = ratedata.sets
            elif isinstance(ratedata, TabularRateData):
                self.original_source = ratedata.original_source                
                self.chapter = ratedata.chapter
                self.reactants = ratedata.reactants
                self.products = ratedata.products
                self.table_file = ratedata.table_file
                self.table_header_lines = ratedata.table_header_lines
                self.table_rhoy_lines = ratedata.table_rhoy_lines
                self.table_temp_lines = ratedata.table_temp_lines
                self.table_num_vars = ratedata.table_num_vars
                self.table_index_name = ratedata.table_index_name
            # Calculate rate properties
            self.compute_prefactor()
            self.decide_screening()
            self.compose_strings()

    def __repr__(self):
        return self.string

    @staticmethod
    def get_rate_file_path(rate_file):
        # get the rates in the list of rate files
        pyreaclib_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 
        pyreaclib_rates_dir = os.path.join(pyreaclib_dir, 'rates')
        filepath = None
        # check to see if the rate file is in the working dir
        fp = glob.glob(rate_file)
        if fp:
            filepath = fp[0]
        else:
            # check to see if the rate file is in pyreaclib/rates
            fp = glob.glob(os.path.join(pyreaclib_rates_dir, rate_file))
            if fp:
                filepath = fp[0]
            else: # Notify of missing file before exiting
                print('ERROR: File {} not found in {} or the working directory!'.format(
                    rate_file, pyreaclib_rates_dir))
        return filepath
    
    @staticmethod
    def parse_rate_file(ratefile):
        """ 
        Parse a standalone file containing a single rate.
        File should be in Reaclib v2 format though
        the data can be any Reaclib version.
        """
        rfpath = Rate.get_rate_file_path(ratefile)
        # read in the file and close
        f = open(rfpath, "r")
        lines = f.readlines()
        f.close()

        # extract rate data from lines
        ratedata,_ = Rate.rate_from_lines(lines)
        return ratedata

    @staticmethod
    def line_is_chapter(line):
        """ 
        Given line, a single string, return
        True if line corresponds to a chapter
        and False otherwise.
        """
        if re.fullmatch('\A[0-9]{1,2}|t\Z', line.strip()):
            return True
        else:
            return False
        
    @staticmethod
    def get_chapter(lines):
        # lines is a list of strings.
        # Get the chapter - lone integer or 't'
        # on a line preceding the other line data
        # Also returns the remaining lines
        llines = lines[:]
        nlines = len(llines)
        for i in range(nlines):
            iline = llines.pop(0)
            if Rate.line_is_chapter(iline):
                chapter = iline.strip()
                break
        return chapter, llines

    @staticmethod
    def rate_from_lines(lines):
        """
        Given lines which define the rate,
        extract the rate information.

        This function will extract as many sets
        as necessary for a single rate corresponding
        to the first set.

        This function returns the rate data and 
        remaining lines which do not correspond to this rate.

        'None' will be returned for the rate data if
        no rate data could be extracted.
        """
        chapter = None
        reactants = []
        products = []
        this_rate_data = None
        # Get the chapter
        chapter,_ = Rate.get_chapter(lines)

        # catch table prescription
        if chapter != "t":
            try:
                chapter = int(chapter)
            except:
                print('ERROR: unrecognized chapter identifier!')
                exit()

        # remove any black lines
        set_lines = [l for l in lines[:] if not l.strip() == ""]

        if chapter == "t":
            # e1 -> e2, Tabulated
            try:
                s0 = set_lines.pop(0)
                s1 = set_lines.pop(0)
                s2 = set_lines.pop(0)
                s3 = set_lines.pop(0)
                s4 = set_lines.pop(0)
                s5 = set_lines.pop(0)
                f = s1.split()
                reactants.append(Nucleus(f[0]))
                products.append(Nucleus(f[1]))

                table_file = s2.strip()
                table_header_lines = int(s3.strip())
                table_rhoy_lines   = int(s4.strip())
                table_temp_lines   = int(s5.strip())
                # Hard-coded number of variables in tables for now.
                table_num_vars     = 6 
                table_index_name = 'j_{}_{}'.format(reactants[0],
                                                    products[0])
                original_source = "".join([s0, s1, s2, s3, s4, s5])
                this_rate_data = TabularRateData(original_source,
                                                 chapter,
                                                 reactants,
                                                 products,
                                                 table_file,
                                                 table_header_lines,
                                                 table_rhoy_lines,
                                                 table_temp_lines,
                                                 table_num_vars,
                                                 table_index_name)
            except:
                pass
        else:
            # the rest is the sets
            Q = None
            first = True
            sets = []
            different_rate = False
            original_source = ""
            while len(set_lines) > 3 and not different_rate:
                # sets are 4 lines long
                s0 = set_lines[0] # chapter
                s1 = set_lines[1]
                s2 = set_lines[2]
                s3 = set_lines[3]
                print(s0)
                print(s1)
                print(s2)
                print(s3)
                # first line of a set has up to 6 nuclei, then the label,
                # and finally the Q value
                f = s1.split()
                Q = f.pop()
                label = f.pop()

                if first:
                    # This is the first set - get reaction info
                    reactants, products = Rate.get_react_prod(f, chapter)
                    first = False
                else:
                    # This is not the first set, so make sure this
                    # set corresponds to the same reaction the
                    # first set describes, otherwise stop reading sets.
                    sreactants, sproducts = Rate.get_react_prod(f, chapter)
                    if not (set(reactants) == set(sreactants) and
                            set(products)  == set(sproducts) and
                            int(s0.strip()) == chapter):
                        # This is not the same reaction as the first set
                        # Stop reading set_lines and return what we have
                        different_rate = True
                if not different_rate:
                    # the second line contains the first 4 coefficients
                    # the third lines contains the final 3
                    # we can't just use split() here, since the fields run into one another
                    n = 13  # length of the field
                    a = [s2[i:i+n] for i in range(0, len(s2), n)]
                    a += [s3[i:i+n] for i in range(0, len(s3), n)]

                    a = [float(e) for e in a if not e.strip() == ""]
                    sets.append(SingleSet(a, label=label))
                    original_source += "".join([s0, s1, s2, s3])
                    # Pop the 1 chapter and 3 set lines we just used from set_lines
                    set_lines.pop(0)
                    set_lines.pop(0)
                    set_lines.pop(0)
                    set_lines.pop(0)
            if Q:
                this_rate_data = ReaclibRateData(original_source,
                                                 chapter,
                                                 reactants,
                                                 products,
                                                 Q,
                                                 sets)
        return this_rate_data, set_lines

    @staticmethod
    def get_react_prod(f, chapter):
        # f is a list specifying the nuclei -- their interpretation
        # depends on the chapter
        reactants = []
        products  = []
        if chapter == 1:
            # e1 -> e2
            reactants.append(Nucleus(f[0]))
            products.append(Nucleus(f[1]))

        elif chapter == 2:
            # e1 -> e2 + e3
            reactants.append(Nucleus(f[0]))
            products += [Nucleus(f[1]), Nucleus(f[2])]

        elif chapter == 3:
            # e1 -> e2 + e3 + e4
            reactants.append(Nucleus(f[0]))
            products += [Nucleus(f[1]), Nucleus(f[2]), Nucleus(f[3])]

        elif chapter == 4:
            # e1 + e2 -> e3
            reactants += [Nucleus(f[0]), Nucleus(f[1])]
            products.append(Nucleus(f[2]))

        elif chapter == 5:
            # e1 + e2 -> e3 + e4
            reactants += [Nucleus(f[0]), Nucleus(f[1])]
            products += [Nucleus(f[2]), Nucleus(f[3])]

        elif chapter == 6:
            # e1 + e2 -> e3 + e4 + e5
            reactants += [Nucleus(f[0]), Nucleus(f[1])]
            products += [Nucleus(f[2]), Nucleus(f[3]), Nucleus(f[4])]

        elif chapter == 7:
            # e1 + e2 -> e3 + e4 + e5 + e6
            reactants += [Nucleus(f[0]), Nucleus(f[1])]
            products += [Nucleus(f[2]), Nucleus(f[3]),
                              Nucleus(f[4]), Nucleus(f[5])]

        elif chapter == 8:
            # e1 + e2 + e3 -> e4
            reactants += [Nucleus(f[0]), Nucleus(f[1]), Nucleus(f[2])]
            products.append(Nucleus(f[3]))

        elif chapter == 9:
            # e1 + e2 + e3 -> e4 + e5
            reactants += [Nucleus(f[0]), Nucleus(f[1]), Nucleus(f[2])]
            products += [Nucleus(f[3]), Nucleus(f[4])]

        elif chapter == 10:
            # e1 + e2 + e3 + e4 -> e5 + e6
            reactants += [Nucleus(f[0]), Nucleus(f[1]),
                               Nucleus(f[2]), Nucleus(f[3])]
            products += [Nucleus(f[4]), Nucleus(f[5])]

        elif chapter == 11:
            # e1 -> e2 + e3 + e4 + e5
            reactants.append(Nucleus(f[0]))
            products += [Nucleus(f[1]), Nucleus(f[2]),
                              Nucleus(f[3]), Nucleus(f[4])]
            
        return reactants, products

    def compose_strings(self):
        self.string = ""
        self.pretty_string = r"$"
        for n, r in enumerate(self.reactants):
            self.string += "{}".format(r)
            self.pretty_string += r"{}".format(r.pretty)
            if not n == len(self.reactants)-1:
                self.string += " + "
                self.pretty_string += r" + "

        self.string += " --> "
        self.pretty_string += r" \rightarrow "

        for n, p in enumerate(self.products):
            self.string += "{}".format(p)
            self.pretty_string += r"{}".format(p.pretty)
            if not n == len(self.products)-1:
                self.string += " + "
                self.pretty_string += r" + "

        self.pretty_string += r"$"

    def compute_prefactor(self):
        """ compute statistical prefactor and density exponent from the reactants """
        self.prefactor = 1.0  # this is 1/2 for rates like a + a (double counting)
        self.inv_prefactor = 1
        for r in set(self.reactants):
            self.inv_prefactor = self.inv_prefactor * np.math.factorial(self.reactants.count(r))
        self.prefactor = self.prefactor/float(self.inv_prefactor)
        self.dens_exp = len(self.reactants)-1

    def decide_screening(self):
        """ determine if this rate is eligible for screening """
        nucz = []
        for parent in self.reactants:
            if parent.Z != 0:
                nucz.append(parent)
        if len(nucz) > 1:
            nucz.sort(key=lambda x: x.Z)
            self.ion_screen = []
            self.ion_screen.append(nucz[0])
            self.ion_screen.append(nucz[1])
        
    def eval(self, T):
        """ evauate the reaction rate for temperature T """
        tf = Tfactors(T)
        r = 0.0
        for s in self.sets:
            f = s.f()
            r += f(tf)

        return r

    def get_rate_exponent(self, T0):
        """
        for a rate written as a power law, r = r_0 (T/T0)**nu, return
        nu corresponding to T0
        """

        # nu = dln r /dln T, so we need dr/dT
        r1 = self.eval(T0)
        dT = 1.e-8*T0
        r2 = self.eval(T0 + dT)

        drdT = (r2 - r1)/dT
        return (T0/r1)*drdT

    def plot(self, Tmin=1.e7, Tmax=1.e10):

        T = np.logspace(np.log10(Tmin), np.log10(Tmax), 100)
        r = np.zeros_like(T)

        for n in range(len(T)):
            r[n] = self.eval(T[n])

        plt.loglog(T, r)

        plt.xlabel(r"$T$")

        if self.dens_exp == 0:
            plt.ylabel(r"\tau")
        elif self.dens_exp == 1:
            plt.ylabel(r"$N_A <\sigma v>$")
        elif self.dens_exp == 2:
            plt.ylabel(r"$N_A^2 <n_a n_b n_c v>$")

        plt.title(r"{}".format(self.pretty_string))

        plt.show()
        
