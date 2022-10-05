import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from mymi import types

# Define region color map.
palette_tab20 = plt.cm.tab20
palette_tab20b = plt.cm.tab20b
class RegionColours:
    BrachialPlexus_L = palette_tab20(0)
    BrachialPlexus_R = palette_tab20(1)
    Brain = palette_tab20(14)
    BrainStem = palette_tab20(15)
    Cochlea_L = palette_tab20(10)
    Cochlea_R = palette_tab20(11)
    Lens_L = palette_tab20(6)
    Lens_R = palette_tab20(7)
    Mandible = palette_tab20(16)
    OpticNerve_L = palette_tab20(8)
    OpticNerve_R = palette_tab20(9)
    OralCavity = palette_tab20(16)
    Parotid_L = palette_tab20(4)
    Parotid_R = palette_tab20(5)
    SpinalCord = palette_tab20b(0)
    Submandibular_L = palette_tab20(12)
    Submandibular_R = palette_tab20(13)

def to_255(colour: types.Colour) -> Tuple[int, int, int]:
    """
    returns: a colour in RGB (0-255) scale.
    args:
        colour: the colour in RGB (0-1) scale.
    """
    # Convert colour scale.
    colour = tuple((255 * np.array(colour)).astype(int))
    return colour