import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Dict

def force_of_gravity(mass1: float, mass2: float, r: np.ndarray) -> np.ndarray:
    ''' Returns force on mass2 due to mass1 (single r vector)
    
        Args:
            mass1: mass of planet 1
            mass2: mass of planet 2
            r:     vector from planet 1 to planet 2
    '''
    G = 6.67408e-11
    return ((- G * mass1 * mass2) / np.linalg.norm(r)**3) * r

def separate_force_of_gravity(mass1: float, mass2: float, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    ''' Returns force on mass2 due to mass1 (separate r vectors)
    
        Args:
            mass1: mass of planet 1
            mass2: mass of planet 2
            r1:    position vector of planet 1
            r2:    position vector of planet 2
    '''
    r = r2 - r1
    return force_of_gravity(mass1, mass2, r)

def gravity_force_vector(mass1: float, mass2: float, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    return np.transpose(np.concatenate((
        [r2], 
        [r2 + separate_force_of_gravity(mass1, mass2, r1, r2)]
        )))

def draw_planet(x: float, y: float, z: float, name: str = None, color: str = None) -> go.Scatter3d:
    ''' Returns plotly Scatter3d object for a planet
    
        Args:
            x:     x position of planet
            y:     y position of planet
            z:     z position of planet
            name:  name of planet
            color: color to draw planet marker
    '''
    return go.Scatter3d(
        x = [x],
        y = [y],
        z = [z],
        mode = 'markers',
        name = name if name else None,
        marker = dict(color = color) if color else None
    )

def draw_force_line(
    force_x: np.ndarray, force_y: np.ndarray, 
    force_z: np.ndarray, name: str = None, color: str = None) -> go.Scatter3d:
    ''' Returns plotly Scatter3d object for a force line
                                    (line part of vector)

        Args:
            force_x: x coords of two endpoints 
                     of force vector
            force_y: y coords of two endpoints 
                     of force vector
            force_z: z coords of two endpoints 
                     of force vector
            color:   color to draw force line
    '''
    return go.Scatter3d(
        x = force_x, y = force_y, z = force_z,
        mode = 'lines',
        name = name if name else None,
        showlegend = False,
        line = dict(
            width = 10,
            color = color if color else None
        )
    )

def draw_force_cone(
    force_x: np.ndarray, force_y: np.ndarray, force_z: np.ndarray, 
    name: str = None, color: str = None, anchor: str = "tip") -> go.Cone:
    ''' Returns plotly Cone object for a force cone
                            (arrow part of vector)

        Args:
            force_x: x coords of two endpoints 
                     of force vector
            force_y: y coords of two endpoints 
                     of force vector
            force_z: z coords of two endpoints
                    of force vector
            color:   color to draw cone
    '''
    force = [f[1] - f[0] for f in [force_x, force_y, force_z]]
    return go.Cone(
        x = [force_x[1]], y = [force_y[1]], z = [force_z[1]],
        u = [force[0]], v = [force[1]], w = [force[2]],
        sizemode = 'scaled',
        name = name if name else None,
        sizeref = 0.10,
        showscale = False,
        autocolorscale = False,
        showlegend = False,
        anchor=anchor,
        colorscale = [[0, color], [1, color]] if name else None
    )

def ranger(coord1: float, coord2: float, force_on_1: np.ndarray, force_on_2: np.ndarray) -> Tuple[float]:
    ''' Returns tuple of (min, max) value from given arguments
        min can also contain 0
        
        All coords in 1 dimension only
    
        Args:
            coord1:     coordinate for planet 1
            coord2:     coordinate for planet 2
            force_on_1: coords of two endpoints of force
                        vector on planet 1
            force_on_2: coords of two endpoints of force
                        vector on planet 2
    '''
    return (
        min(np.concatenate((
            [0.], [coord1], [coord2], force_on_1, force_on_2
        ))),
        max(np.concatenate((
            [coord1], [coord2], force_on_1, force_on_2
        )))
    )

def set_ranges(*position_vectors: np.ndarray) -> Dict[str, Dict[str, Tuple[float]]]:
    xes = np.concatenate(([0.], np.transpose(position_vectors)[0]))
    yes = np.concatenate(([0.], np.transpose(position_vectors)[1]))
    zes = np.concatenate(([0.], np.transpose(position_vectors)[2]))
    return dict(
        xaxis = dict(range = (min(xes), max(xes))),
        yaxis = dict(range = (min(yes), max(yes))),
        zaxis = dict(range = (min(zes), max(zes)))
    )

# plotly axis configuration data
axes_config = dict(
    showbackground = False,
    zerolinecolor = 'black',
    spikecolor = 'lawngreen',
    gridcolor = 'gainsboro',
    zerolinewidth = 5,
)

plotly_layout = dict(
    scene = dict(
        xaxis = axes_config,
        yaxis = axes_config,
        zaxis = axes_config
    ),
    margin = dict(
        r = 10, l = 10, b = 0, t = 0
    ),
    scene_camera = dict(
        eye = dict(
            x = 1,
            y = 2.25,
            z = 1
        )
    )
)

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']