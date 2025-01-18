import numpy as np
import numexpr as ne
from numba import njit, jit, generated_jit
import itertools as it
from tqdm.notebook import tqdm

G = 4 * np.pi ** 2

def accel_grav(mass, position, velocity=None, grav_const=G):
    assert mass.shape[0] == position.shape[0]
    num_objs = position.shape[0]
    accel_over_G_result = np.zeros(position.shape)
    pairs = np.triu_indices(n=num_objs, k=1)

    # Newton's gravitational force on mass i due to j: 
    #   F_ij = G m_i m_j (r_j - r_i) / (r_ij ** 3)
    #   => a_i = G m_j (r_j - r_i) / (r_ij ** 3)
    disp_pairs = position[pairs[1]] - position[pairs[0]]
    dist_cubed_pairs = (disp_pairs**2).sum(axis=1)**1.5
    disp_over_dist_cubed_pairs = \
        disp_pairs / dist_cubed_pairs[..., np.newaxis].repeat(3, axis=1)
    accel_over_G_m = np.zeros((num_objs, *accel_over_G_result.shape))
    accel_over_G_m[pairs] = disp_over_dist_cubed_pairs
    accel_over_G_m -= np.transpose(accel_over_G_m, axes=(1, 0, 2))
    accel_over_G_result = np.tensordot(accel_over_G_m, mass, axes=((1), (0)))
    return grav_const * accel_over_G_result

def accel_intermolecular(mass, position, velocity=None, V_LJ=1, r_m=1):
    assert mass.shape[0] == position.shape[0]
    num_objs = position.shape[0]
    accel_over_V_result = np.zeros(position.shape)
    pairs = np.triu_indices(n=num_objs, k=1)

    # Newton's gravitational force on mass i due to j: 
    #   F_ij = G m_i m_j (r_j - r_i) / (r_ij ** 3)
    #   => a_i = G m_j (r_j - r_i) / (r_ij ** 3)
    disp_pairs = position[pairs[1]] - position[pairs[0]]
    dist_squared_pairs = (
        ne.evaluate('sum(disp_pairs**2, 1)')[..., np.newaxis].repeat(3, axis=1))
    coeff = ne.evaluate('(r_m ** 6 / dist_squared_pairs ** 3) \
           - (r_m ** 12 / dist_squared_pairs ** 6)')
    disp_over_dist_squared_pairs = ne.evaluate('coeff * disp_pairs / dist_squared_pairs')
    force_over_V = np.zeros((num_objs, *accel_over_V_result.shape))
    force_over_V[pairs] = disp_over_dist_squared_pairs
    force_over_V -= np.transpose(force_over_V, axes=(1, 0, 2))
    mass_ext = mass[..., np.newaxis]
    accel_over_V_result = ne.evaluate('sum_f / mass_ext', {'sum_f': ne.evaluate('sum(force_over_V, 1)'), 'mass_ext': mass_ext})
    return ne.evaluate('12 * V_LJ * accel_over_V_result')

def accel_softwalls_np(mass, position, velocity, dt, dt_per_period=100,
                    walls=((-8, 8), (-8, 8), (-8, 8))):
    p_below_x = position[:, 0] < walls[0][0]
    p_above_x = position[:, 0] > walls[0][1]
    p_below_y = position[:, 1] < walls[1][0]
    p_above_y = position[:, 1] > walls[1][1]
    p_below_z = position[:, 2] < walls[2][0]
    p_above_z = position[:, 2] > walls[2][1]
    k = (np.pi * 2  / (dt_per_period * dt)) ** 2
    disp = np.zeros(position.shape)
    disp[p_below_x, 0] = position[p_below_x, 0] - walls[0][0]
    disp[p_above_x, 0] = position[p_above_x, 0] - walls[0][1]
    disp[p_below_y, 1] = position[p_below_y, 1] - walls[1][0]
    disp[p_above_y, 1] = position[p_above_y, 1] - walls[1][1]
    disp[p_below_z, 2] = position[p_below_z, 2] - walls[2][0]
    disp[p_above_z, 2] = position[p_above_z, 2] - walls[2][1]
    return -k * disp

def accel_softwalls(mass, position, velocity, dt, dt_per_period=100,
                    walls=((-8, 8), (-8, 8), (-8, 8))):
    p_below_x = position[:, 0] < walls[0][0]
    p_above_x = position[:, 0] > walls[0][1]
    p_below_y = position[:, 1] < walls[1][0]
    p_above_y = position[:, 1] > walls[1][1]
    p_below_z = position[:, 2] < walls[2][0]
    p_above_z = position[:, 2] > walls[2][1]
    k = (np.pi * 2  / (dt_per_period * dt)) ** 2
    disp = np.zeros(position.shape)
    disp[p_below_x, 0] = ne.evaluate('p - w', {'p': position[p_below_x, 0], 'w': walls[0][0]})
    disp[p_above_x, 0] = ne.evaluate('p - w', {'p': position[p_above_x, 0], 'w': walls[0][1]})
    disp[p_below_y, 1] = ne.evaluate('p - w', {'p': position[p_below_y, 1], 'w': walls[1][0]})
    disp[p_above_y, 1] = ne.evaluate('p - w', {'p': position[p_above_y, 1], 'w': walls[1][1]})
    disp[p_below_z, 2] = ne.evaluate('p - w', {'p': position[p_below_z, 2], 'w': walls[2][0]})
    disp[p_above_z, 2] = ne.evaluate('p - w', {'p': position[p_above_z, 2], 'w': walls[2][1]})
    return ne.evaluate('-k * disp')

@njit
def first_step(position, velocity, acceleration, dt):
    return position + velocity * dt + 0.5 * acceleration * dt ** 2

def first_step_ne(position, velocity, acceleration, dt):
    return ne.evaluate('position + velocity * dt + 0.5 * acceleration * dt ** 2')

@njit
def stormer_verlet_step(position, position_old, acceleration, 
                        dt, dt_old=None):
    if dt_old is None: dt_old = dt
    return position + (position - position_old) * dt / dt_old \
           + 0.5 * acceleration * (dt + dt_old) * dt

def stormer_verlet_step_ne(position, position_old, acceleration, 
                        dt, dt_old=None):
    if dt_old is None: dt_old = dt
    return ne.evaluate('position + (position - position_old) * dt / dt_old \
           + 0.5 * acceleration * (dt + dt_old) * dt')

def vel_verlet_step(mass, position, velocity, 
                    dt, dt_old=None, accel_func=None):
    if dt_old is None: dt_old = dt
    if accel_func is None: accel_func = lambda mass, position, velocity, dt: 0
    velocity_halfstep = velocity + 0.5 * accel_func(mass=mass, 
                                                    position=position, 
                                                    velocity=velocity, 
                                                    dt=dt) * dt
    position_new = position + velocity_halfstep * dt
    acceleration_new = accel_func(mass=mass, 
                                  position=position_new, 
                                  velocity=velocity_halfstep,
                                  dt=dt)
    velocity_new = velocity_halfstep + 0.5 * acceleration_new * dt
    return position_new, velocity_new

@njit
def stormer_verlet_vel(position_new, position_old, dt, dt_old=None):
    if dt_old is None: dt_old = dt
    return (position_new - position_old) / (dt + dt_old)

def stormer_verlet_vel_ne(position_new, position_old, dt, dt_old=None):
    if dt_old is None: dt_old = dt
    return ne.evaluate('(position_new - position_old) / (dt + dt_old)')

def accel_box(mass, position, velocity, dt):
    accel1 = accel_intermolecular(
        mass=mass, position=position, velocity=velocity, V_LJ=1, r_m=1
    ) 
    accel2 = accel_softwalls(
        mass=mass, position=position, velocity=velocity, dt=dt, 
        dt_per_period=8, walls=((-8, 8), (-8, 8), (-8, 8))
    )
    return accel1 + accel2

def integrate_verlet(mass, initial_position, initial_velocity, 
                     total_time, dt, 
                     accel_func=accel_box, is_accel_vel_dependent=False):

    # Determine if velocity is needed for acceleration
    vel_for_init_accel = None if is_accel_vel_dependent else initial_velocity
    if is_accel_vel_dependent:
        vel_func = lambda i: None
    else:
        vel_func = lambda i: vel_verlet_step(
            mass, 
            positions_over_time[i - 2], 
            velocities_over_time[i - 2], 
            dt, 
            dt_old=None, 
            accel_func=accel_func)[1]
    
    num_steps = np.ceil(total_time / dt).astype(int)

    # Initializing positions and velocities over time arrays
    positions_over_time = np.zeros((num_steps + 1, *initial_position.shape))
    velocities_over_time = np.zeros(positions_over_time.shape)
    
    # Initial positions and velocities placed in arrays
    positions_over_time[0] = initial_position
    velocities_over_time[0] = initial_velocity

    # First step's positions and velocities
    positions_over_time[1] = first_step(
        position=initial_position, 
        velocity=initial_velocity, 
        acceleration=accel_func(
            mass=mass, 
            position=initial_position, 
            velocity=vel_for_init_accel,
            dt=dt),
        dt=dt)


    # First step's velocities will be updated alongside the next iterations 
    # of verlet steps   
    pbar = tqdm(total=num_steps + 1) # Init pbar
    num_pbar__intervals = int(num_steps / 100) if num_steps > 100 else 1
    for i in range(2, num_steps + 1):
        positions_over_time[i] = stormer_verlet_step(
            position=positions_over_time[i - 1], 
            position_old=positions_over_time[i - 2], 
            acceleration=accel_func(
                mass=mass, 
                position=positions_over_time[i - 1],
                velocity=vel_func(i),
                dt=dt), 
            dt=dt, dt_old=None)
        velocities_over_time[i - 1] = stormer_verlet_vel(
            position_new=positions_over_time[i],
            position_old=positions_over_time[i - 2],
            dt=dt, dt_old=None)
        
        if num_steps - i < num_steps % num_pbar__intervals: pbar.update(n=1)
        elif (i+1) % num_pbar__intervals == 0: 
            pbar.update(n=num_pbar__intervals)
        
    
    # Perform one extra stormer verlet step to get the last velocity
    position_extra = stormer_verlet_step(
        position=positions_over_time[-1], 
        position_old=positions_over_time[-2], 
        acceleration=accel_func(
            mass=mass, 
            position=positions_over_time[-1],
            velocity=vel_func(num_steps + 1),
            dt=dt), 
        dt=dt, dt_old=None)
    velocities_over_time[-1] = stormer_verlet_vel(
        position_new=position_extra,
        position_old=positions_over_time[-1],
        dt=dt, dt_old=None)
    pbar.update(n=1)
    return positions_over_time, velocities_over_time


# mass = np.array([1, 2, 2, 1])
# position = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
# acceleration = accel_grav(mass, position)
# print(acceleration)
